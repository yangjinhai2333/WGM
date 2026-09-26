"""Standalone evaluation journal. No runners, schedulers or algorithm state.

Writers must hold a Session across marker publication AND the external action.
Only a newly published marker (Publication.created) permits that action. A
replayed marker is a no-op, never permission to launch again.
"""
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields, replace
from enum import Enum
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
import time


SCHEMA = 2
READ_SCHEMAS = (1, 2)
MAX_JSON_BYTES = 1024 * 1024
DIRECTORY = "evaluation_ledger"
STAGES = ("root", "search", "checkpoint_bai", "final_benchmark")
KINDS = ("planned", "reserved", "preparation_started", "preparation_terminal",
         "model_started", "model_terminal", "model_result_ready",
         "evaluation_started", "evaluation_terminal", "bound")
AUX_KINDS = ("model_start_confirmed", "evaluation_start_confirmed", "cleanup_terminal")
RECOVERY_KINDS = ("attempt_abandoned",)
CONNECTION_RETRY_REASON = "audited_connection_failure_retry"
ERROR_CODES = ("IO", "PREPARATION_FAILED", "INTERRUPTED", "TIMEOUT", "READINESS_RECEIPT_REJECTED",
               "MODEL_FAILED", "EVALUATOR_FAILED", "ARTIFACT_INVALID", "CLEANUP_FAILED")


class LedgerError(ValueError):
    """Fixed codes only: never echo input, exception text or paths."""

    def __init__(self, code):
        self.code = code
        super().__init__(code)


def _require(ok, code):
    if not ok:
        raise LedgerError(code)


@contextmanager
def _io_errors():
    try:
        yield
    except OSError:
        raise LedgerError("LEDGER_IO") from None


def _id(value):
    _require(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value),
             "LEDGER_ID_INVALID")


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _sha(value):
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "LEDGER_DIGEST_INVALID")


def canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, RecursionError):
        raise LedgerError("LEDGER_JSON_INVALID") from None


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "LEDGER_JSON_DUPLICATE")
        result[key] = value
    return result


def _decode(data):
    try:
        _require(type(data) in (str, bytes), "LEDGER_JSON_INVALID")
        _require(len(data.encode("utf-8") if type(data) is str else data) <= MAX_JSON_BYTES, "LEDGER_JSON_SIZE")
        return json.loads(data, object_pairs_hook=_pairs,
                          parse_float=_finite_float,
                          parse_constant=lambda _: (_require(False, "LEDGER_JSON_INVALID")))
    except (ValueError, UnicodeError, RecursionError) as error:
        if isinstance(error, LedgerError):
            raise
        raise LedgerError("LEDGER_JSON_INVALID") from None


def _finite_float(value):
    result = float(value)
    _require(math.isfinite(result), "LEDGER_JSON_INVALID")
    return result


def same_json(left, right):
    """Unlike Python equality, canonical bytes distinguish true, 1 and 1.0."""
    return canonical(left) == canonical(right)


def _keys(value, names):
    _require(type(value) is dict and set(value) == set(names), "LEDGER_FIELDS_INVALID")


def _optional_keys(value, required, optional):
    _require(type(value) is dict and set(required) <= set(value) <= set(required) | set(optional),
             "LEDGER_FIELDS_INVALID")


def validate_execution_evidence(data):
    """Whitelist positive terminal evidence and numeric accounting only."""
    _optional_keys(data, ("started", "reaped", "launch_failed", "returncode", "timed_out", "interrupted",
                 "duration_seconds", "input_tokens", "cached_input_tokens", "output_tokens"), ("usage_error_code",))
    _require(data.get("usage_error_code") in (None, "USAGE_WRITE_FAILED"), "LEDGER_EVIDENCE_INVALID")
    for name in ("started", "reaped", "launch_failed", "timed_out", "interrupted"):
        _require(type(data[name]) is bool, "LEDGER_EVIDENCE_INVALID")
    if data["launch_failed"]:
        _require(not data["started"] and not data["reaped"] and data["returncode"] is None
                 and not data["timed_out"] and not data["interrupted"], "LEDGER_EVIDENCE_INVALID")
    else:
        _require(data["started"] and data["reaped"] and type(data["returncode"]) is int,
                 "LEDGER_EVIDENCE_INVALID")
    duration = data["duration_seconds"]
    _require(type(duration) in (int, float) and math.isfinite(duration) and duration >= 0, "LEDGER_EVIDENCE_INVALID")
    for name in ("input_tokens", "cached_input_tokens", "output_tokens"):
        _require(data[name] is None or _integer(data[name]), "LEDGER_EVIDENCE_INVALID")


def _schema(value, expected=None):
    _require(type(value) is int and value in READ_SCHEMAS
             and (expected is None or value == expected), "LEDGER_SCHEMA_UNSUPPORTED")


def _checked_path(path):
    """Reject symlinks at every existing component; never resolve through one."""
    path = Path(path)
    _require(".." not in path.parts, "LEDGER_PATH_INVALID")
    path = path.absolute()
    for component in (*reversed(path.parents), path):
        _require(not component.is_symlink(), "LEDGER_PATH_INVALID")
    return path


def _regular(path):
    path = _checked_path(path)
    _require(path.is_file(), "LEDGER_FILE_MISSING")
    _require(stat.S_ISREG(path.stat().st_mode) and path.stat().st_nlink == 1, "LEDGER_PATH_INVALID")
    return path


def _read_json(path):
    path = _regular(path)
    _require(path.stat().st_size <= MAX_JSON_BYTES, "LEDGER_JSON_SIZE")
    return _decode(path.read_bytes())


def _sync_dir(path):
    fd = os.open(_checked_path(path), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _mkdir(path):
    path = _checked_path(path)
    if not path.exists():
        try:
            path.mkdir(mode=0o700)
        except FileExistsError:
            pass  # Cooperative creators may race before acquiring the lock.
        _sync_dir(path.parent)
    _require(path.is_dir() and not path.is_symlink(), "LEDGER_PATH_INVALID")


def _publish(path, value, *, replace_head=False, fault=lambda point: None, prefix="event"):
    """Only durable_head is replaceable. Official event files are immutable."""
    path = _checked_path(path)
    _require(replace_head or not path.exists(), "LEDGER_EVENT_EXISTS")
    encoded = canonical(value) + b"\n"
    _require(len(encoded) <= MAX_JSON_BYTES, "LEDGER_JSON_SIZE")
    fault(prefix + ".before_write")
    fd, temporary = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    # Leave abandoned temporary files as crash evidence; readers ignore them.
    with os.fdopen(fd, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        fault(prefix + ".after_write")
        os.fsync(stream.fileno())
        fault(prefix + ".after_file_fsync")
    _require(replace_head or not path.exists(), "LEDGER_EVENT_EXISTS")
    os.rename(temporary, path)  # All cooperative writers hold the same flock.
    fault(prefix + ".after_rename")
    _sync_dir(path.parent)
    fault(prefix + ".after_dir_fsync")


@contextmanager
def _exclusive(path):
    with _io_errors():
        path = _checked_path(path)
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            _regular(path)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise LedgerError("LEDGER_LOCK_BUSY") from None
            os.fsync(fd)
            _sync_dir(path.parent)
            yield
        finally:
            os.close(fd)  # Never unlink a lock inode; another owner may wait on it.


@dataclass(frozen=True)
class Identity:
    run_id: str
    evaluation_id: str
    stage: str
    task_id: str
    task_draw_index: int | None
    candidate_id: str
    task_input_fingerprint: str
    model_fingerprint: str
    policy_fingerprint: str
    source_fingerprint: str
    evaluator_fingerprint: str
    ordinal: int
    schema_version: int = SCHEMA

    def __post_init__(self):
        _schema(self.schema_version)
        for name in ("run_id", "evaluation_id", "task_id", "candidate_id"):
            _id(getattr(self, name))
        _require(self.stage in STAGES and _integer(self.ordinal), "LEDGER_IDENTITY_INVALID")
        _require(self.task_draw_index is None or _integer(self.task_draw_index), "LEDGER_IDENTITY_INVALID")
        for field in fields(self):
            if field.name.endswith("_fingerprint"):
                _sha(getattr(self, field.name))


def _identity(data):
    _keys(data, (f.name for f in fields(Identity)))
    return Identity(**data)


@dataclass(frozen=True)
class Failure:
    code: str
    errno: int | None = None
    cleanup_code: str | None = None

    def __post_init__(self):
        _require(self.code in ERROR_CODES and (self.errno is None or _integer(self.errno)),
                 "LEDGER_ERROR_INVALID")
        _require(self.cleanup_code is None or self.cleanup_code in ERROR_CODES, "LEDGER_ERROR_INVALID")


@dataclass(frozen=True)
class Artifact:
    path: str  # Relative to this evaluation, always artifacts/<attempt>/<name>.
    sha256: str
    size: int

    def __post_init__(self):
        _require(isinstance(self.path, str), "LEDGER_PATH_INVALID")
        parts = self.path.split("/")
        _require(len(parts) == 3 and parts[0] == "artifacts", "LEDGER_PATH_INVALID")
        for part in parts[1:]:
            _id(part)
        _sha(self.sha256)
        _require(_integer(self.size), "LEDGER_ARTIFACT_INVALID")


def _artifacts(data, attempt_id, *, allow_empty=False):
    _require(type(data) is list and (bool(data) or allow_empty), "LEDGER_ARTIFACT_INVALID")
    result = []
    for value in data:
        _keys(value, ("path", "sha256", "size"))
        artifact = Artifact(**value)
        _require(artifact.path.split("/")[1] == attempt_id, "LEDGER_ARTIFACT_IDENTITY")
        result.append(artifact)
    _require([r.path for r in result] == sorted({r.path for r in result}), "LEDGER_ARTIFACT_INVALID")
    return tuple(result)


def invocation_id(identity, attempt_id, kind):
    return digest({"run_id": identity.run_id, "evaluation_id": identity.evaluation_id,
                   "attempt_id": attempt_id, "kind": kind})


@dataclass(frozen=True)
class Event:
    sequence: int
    kind: str
    attempt_id: str | None
    data: bytes  # Canonical immutable JSON; no shared mutable dictionaries.
    sha256: str
    created_at_ns: int


@dataclass(frozen=True)
class Attempt:
    attempt_id: str
    state: str = "reserved"
    preparation_status: str | None = None
    preparation_artifacts: tuple[Artifact, ...] = ()
    model_start_confirmed: bool = False
    evaluation_start_confirmed: bool = False
    cleanup_status: str | None = None
    call_id: str | None = None
    model_status: str | None = None
    model_error: Failure | None = None
    model_terminal_digest: str | None = None
    model_artifacts: tuple[Artifact, ...] = ()
    evaluator_call_id: str | None = None
    evaluation_status: str | None = None
    evaluation_error: Failure | None = None
    result_artifacts: tuple[Artifact, ...] = ()
    evaluation_terminal_digest: str | None = None
    authorization_artifacts: tuple[Artifact, ...] = ()


@dataclass(frozen=True)
class Snapshot:
    identity: Identity
    events: tuple[Event, ...]
    attempts: tuple[Attempt, ...]

    @property
    def state(self):
        return self.attempts[-1].state if self.attempts else "planned"

    @property
    def binding_id(self):
        return next((e.sha256 for e in reversed(self.events) if e.kind == "bound"), None)


def _terminal(data):
    _require(data["status"] in ("success", "failed"), "LEDGER_TERMINAL_INVALID")
    if data["status"] == "success":
        _require(data["error"] is None, "LEDGER_TERMINAL_INVALID")
    else:
        _keys(data["error"], ("code", "errno", "cleanup_code"))
        Failure(**data["error"])


def reduce_events(rows, *, schema_version=SCHEMA):
    """Pure hash-chain and state validator. Does not read or change files."""
    _require(bool(rows), "LEDGER_NOT_PLANNED")
    _schema(schema_version)
    identity, previous, events, attempts, seen = None, None, [], [], set()
    for sequence, row in enumerate(rows, 1):
        _keys(row, ("schema_version", "sequence", "previous_digest", "sha256", "created_at_ns",
                    "kind", "attempt_id", "identity_sha256", "data"))
        _schema(row["schema_version"], schema_version)
        _require(type(row["sequence"]) is int and row["sequence"] == sequence, "LEDGER_SEQUENCE")
        _require(row["previous_digest"] == previous, "LEDGER_CHAIN")
        _require(row["sha256"] == digest({k:v for k,v in row.items() if k != "sha256"}), "LEDGER_EVENT_DIGEST")
        _require(_integer(row["created_at_ns"], 1)
                 and row["kind"] in KINDS + (AUX_KINDS + RECOVERY_KINDS if schema_version == 2 else ()), "LEDGER_EVENT_INVALID")
        kind, attempt_id, data = row["kind"], row["attempt_id"], row["data"]
        if sequence == 1:
            _require(kind == "planned" and attempt_id is None, "LEDGER_TRANSITION")
            _keys(data, ("identity",))
            identity = _identity(data["identity"])
            _schema(identity.schema_version, schema_version)
        else:
            _id(attempt_id)
            _require((kind, attempt_id) not in seen and kind != "planned", "LEDGER_TRANSITION")
            _require(not attempts or attempts[-1].state != "bound" or kind == "cleanup_terminal"
                     or (kind == "attempt_abandoned" and data.get("reason") == CONNECTION_RETRY_REASON),
                     "LEDGER_ALREADY_BOUND")
            if kind == "reserved":
                _keys(data, ())
                _require(attempt_id not in {a.attempt_id for a in attempts}, "LEDGER_ATTEMPT_REUSED")
                _require(not attempts or (attempts[-1].state == "preparation_terminal"
                         and attempts[-1].preparation_status == "failed") or (
                         attempts[-1].state == 'attempt_abandoned' and attempts[-1].cleanup_status == 'success'),
                         "LEDGER_RETRY_FORBIDDEN")
                attempts.append(Attempt(attempt_id))
            else:
                _require(bool(attempts) and attempts[-1].attempt_id == attempt_id, "LEDGER_ATTEMPT_INVALID")
                a = attempts[-1]
                if kind == "preparation_started":
                    _keys(data, ())
                    _require(a.state == "reserved", "LEDGER_TRANSITION")
                elif kind == "preparation_terminal":
                    _optional_keys(data, ("status", "error"), ("artifacts",) if schema_version == 2 else ())
                    _require(a.state == "preparation_started", "LEDGER_TRANSITION")
                    _terminal(data)
                    a = replace(a, preparation_status=data["status"], preparation_artifacts=
                                _artifacts(data["artifacts"], attempt_id, allow_empty=True) if "artifacts" in data else ())
                elif kind == "model_started":
                    _keys(data, ("call_id",))
                    _require(a.state == "preparation_terminal" and a.preparation_status == "success", "LEDGER_TRANSITION")
                    _require(data["call_id"] == invocation_id(identity, attempt_id, "model"), "LEDGER_CALL_IDENTITY")
                    a = replace(a, call_id=data["call_id"])
                elif kind == "model_terminal":
                    _optional_keys(data, ("status", "error", "call_id", "artifacts"), ("evidence",) if schema_version == 2 else ())
                    _require(a.state == "model_started" and data["call_id"] == a.call_id, "LEDGER_TRANSITION")
                    _terminal(data)
                    if "evidence" in data:
                        validate_execution_evidence(data["evidence"])
                        _require(not a.model_start_confirmed or data["evidence"]["started"], "LEDGER_EVIDENCE_INVALID")
                        if data["status"] == "success":
                            _require(data["evidence"]["returncode"] == 0 and not data["evidence"]["timed_out"]
                                     and not data["evidence"]["interrupted"], "LEDGER_EVIDENCE_INVALID")
                    artifacts = _artifacts(data["artifacts"], attempt_id, allow_empty=True)
                    # Successful output is sealed separately after capture. A
                    # known failure may retain already available safe evidence.
                    _require(data["status"] == "failed" or not artifacts, "LEDGER_ARTIFACT_INVALID")
                    a = replace(a, model_status=data["status"], model_artifacts=artifacts,
                                model_error=Failure(**data["error"]) if data["error"] else None,
                                model_terminal_digest=row["sha256"])
                elif kind == "model_result_ready":
                    _keys(data, ("artifacts",))
                    _require(a.state == "model_terminal" and a.model_status == "success" and not a.model_artifacts,
                             "LEDGER_TRANSITION")
                    a = replace(a, model_artifacts=_artifacts(data["artifacts"], attempt_id))
                elif kind == "evaluation_started":
                    _keys(data, ("evaluator_call_id",))
                    _require(a.state == "model_terminal" and a.model_status == "success"
                             and bool(a.model_artifacts), "LEDGER_TRANSITION")
                    _require(data["evaluator_call_id"] == invocation_id(identity, attempt_id, "evaluator"), "LEDGER_CALL_IDENTITY")
                    a = replace(a, evaluator_call_id=data["evaluator_call_id"])
                elif kind == "evaluation_terminal":
                    _optional_keys(data, ("status", "error", "evaluator_call_id", "artifacts"), ("evidence",) if schema_version == 2 else ())
                    _require(a.state == "evaluation_started" and data["evaluator_call_id"] == a.evaluator_call_id,
                             "LEDGER_TRANSITION")
                    _terminal(data)
                    if "evidence" in data:
                        validate_execution_evidence(data["evidence"])
                        _require(not a.evaluation_start_confirmed or data["evidence"]["started"], "LEDGER_EVIDENCE_INVALID")
                        if data["status"] == "success":
                            _require(not any(data["evidence"][k] for k in ("launch_failed", "timed_out", "interrupted")),
                                     "LEDGER_EVIDENCE_INVALID")
                    artifacts = _artifacts(data["artifacts"], attempt_id, allow_empty=data["status"] == "failed")
                    a = replace(a, evaluation_status=data["status"], result_artifacts=artifacts,
                                evaluation_error=Failure(**data["error"]) if data["error"] else None,
                                evaluation_terminal_digest=row["sha256"])
                elif kind == 'attempt_abandoned':
                    _keys(data, ('artifacts', 'reason'))
                    unknown = (identity.stage == 'search' and a.state == 'model_started'
                             and not a.model_start_confirmed and not a.evaluator_call_id
                             and data['reason'] == 'operator_accepts_unknown_model_outcome')
                    terminal = next((_decode(e.data) for e in reversed(events)
                        if e.kind == 'model_terminal' and e.attempt_id == attempt_id), {})
                    evidence = terminal.get('evidence', {})
                    local = (identity.stage == 'final_benchmark' and a.state == 'model_terminal'
                             and a.model_status == 'failed' and not a.model_start_confirmed
                             and not a.evaluator_call_id and evidence.get('started') is False
                             and evidence.get('launch_failed') is True and not evidence.get('reaped')
                             and a.model_error.code in ('READINESS_RECEIPT_REJECTED', 'INTERRUPTED')
                             and data['reason'] == 'audited_local_readiness_no_start')
                    # Audited retry of a search evaluation whose model call failed on a
                    # connection error: the failed binding stays in the chain for audit.
                    connection = (identity.stage == 'search' and a.state == 'bound'
                                  and a.cleanup_status == 'success' and a.model_status == 'failed'
                                  and not a.evaluator_call_id and evidence.get('started') is True
                                  and evidence.get('reaped') is True and not evidence.get('timed_out')
                                  and not evidence.get('interrupted')
                                  and data['reason'] == CONNECTION_RETRY_REASON)
                    _require(unknown or local or connection, 'LEDGER_RETRY_FORBIDDEN')
                    a = replace(a, authorization_artifacts=_artifacts(data['artifacts'], attempt_id))
                elif kind == "bound":
                    _require(a.state in ("model_terminal", "evaluation_terminal"), "LEDGER_TRANSITION")
                    _require(same_json(data, _binding(identity, a)), "LEDGER_BINDING_INVALID")
                elif kind in ("model_start_confirmed", "evaluation_start_confirmed"):
                    _keys(data, ())
                    _require(a.state == kind.replace("_start_confirmed", "_started"), "LEDGER_TRANSITION")
                    a = replace(a, **{kind: True})
                elif kind == "cleanup_terminal":
                    _keys(data, ("status", "error"))
                    _require(a.state in ("bound", 'attempt_abandoned') or (a.state == "preparation_terminal"
                             and a.preparation_status == "failed"), "LEDGER_TRANSITION")
                    _terminal(data)
                    a = replace(a, cleanup_status=data["status"])
                attempts[-1] = replace(a, state=a.state if kind == "model_result_ready" or kind in AUX_KINDS else kind)
        _require(row["identity_sha256"] == digest(asdict(identity)), "LEDGER_IDENTITY_MISMATCH")
        seen.add((kind, attempt_id))
        events.append(Event(sequence, kind, attempt_id, canonical(data), row["sha256"], row["created_at_ns"]))
        previous = row["sha256"]
    return Snapshot(identity, tuple(events), tuple(attempts))


def _binding(identity, attempt):
    """Derive the only permissible binding from a validated terminal, never a reward."""
    _require(attempt.model_error is None or attempt.model_error.code != 'READINESS_RECEIPT_REJECTED',
             'READINESS_REAUDIT_REQUIRED')
    if attempt.evaluation_status is not None:
        outcome, stage = attempt.evaluation_status, "evaluation"
        terminal, failure = attempt.evaluation_terminal_digest, attempt.evaluation_error
    else:
        _require(attempt.model_status == "failed", "LEDGER_TERMINAL_REQUIRED")
        outcome, stage = "failed", "model"
        terminal, failure = attempt.model_terminal_digest, attempt.model_error
    _require(terminal is not None, "LEDGER_TERMINAL_REQUIRED")
    return {"outcome": outcome, "failure_stage": stage if failure else None,
            "error_code": failure.code if failure else None, "terminal_event_digest": terminal,
            "identity_sha256": digest(asdict(identity)), "attempt_id": attempt.attempt_id,
            "call_id": attempt.call_id, "evaluator_call_id": attempt.evaluator_call_id,
            "artifacts_digest": digest([asdict(a) for a in (*attempt.model_artifacts, *attempt.result_artifacts)])}


class Decision(str, Enum):
    START_NEW = "START_NEW"
    SKIP_BOUND = "SKIP_BOUND"
    START_PREPARATION = "START_PREPARATION"
    RECOVER_PREPARATION = "RECOVER_PREPARATION"
    RETRY_PREPARATION = "RETRY_PREPARATION"
    RETRY_AUTHORIZED = "RETRY_AUTHORIZED"
    START_MODEL = "START_MODEL"
    RECONCILE_MODEL_RESULT = "RECONCILE_MODEL_RESULT"
    RESUME_EVALUATOR = "RESUME_EVALUATOR"
    RECONCILE_EVALUATION_RESULT = "RECONCILE_EVALUATION_RESULT"
    BLOCK_AMBIGUOUS_MODEL_ATTEMPT = "BLOCK_AMBIGUOUS_MODEL_ATTEMPT"
    BLOCK_AMBIGUOUS_EVALUATOR_ATTEMPT = "BLOCK_AMBIGUOUS_EVALUATOR_ATTEMPT"
    RECONCILE_FAILED_MODEL = "RECONCILE_FAILED_MODEL"
    RECONCILE_FAILED_EVALUATION = "RECONCILE_FAILED_EVALUATION"
    BLOCK_CORRUPT_STATE = "BLOCK_CORRUPT_STATE"
    BLOCK_IDENTITY_MISMATCH = "BLOCK_IDENTITY_MISMATCH"
    READ_ONLY = "LEDGER_SCHEMA_READ_ONLY"
    LEGACY = "legacy_without_evaluation_ledger"


@dataclass(frozen=True)
class Inspection:
    snapshot: Snapshot | None = None
    error_code: str | None = None
    legacy: bool = False
    read_only: bool = False


def classify(inspection, expected=None):
    """Pure decision on a read-validated snapshot, with no process inference."""
    if inspection.error_code:
        return (Decision.BLOCK_IDENTITY_MISMATCH if inspection.error_code == "LEDGER_IDENTITY_MISMATCH"
                else Decision.BLOCK_CORRUPT_STATE)
    if inspection.legacy:
        return Decision.LEGACY
    if inspection.read_only or (inspection.snapshot is not None and inspection.snapshot.identity.schema_version == 1):
        return Decision.READ_ONLY
    state = inspection.snapshot
    if state is not None and expected is not None and state.identity != expected:
        return Decision.BLOCK_IDENTITY_MISMATCH
    if state is None or not state.attempts:
        return Decision.START_NEW
    attempt = state.attempts[-1]
    if attempt.state == "bound":
        return Decision.SKIP_BOUND
    if attempt.state == "reserved":
        return Decision.START_PREPARATION
    if attempt.state == 'attempt_abandoned':
        return Decision.RETRY_AUTHORIZED
    if attempt.state == "preparation_started":
        return Decision.RECOVER_PREPARATION
    if attempt.state == "preparation_terminal":
        return Decision.START_MODEL if attempt.preparation_status == "success" else Decision.RETRY_PREPARATION
    if attempt.state == "model_started":
        return Decision.BLOCK_AMBIGUOUS_MODEL_ATTEMPT
    if attempt.state == "model_terminal":
        if attempt.model_status == "failed":
            return Decision.RECONCILE_FAILED_MODEL
        return Decision.RESUME_EVALUATOR if attempt.model_artifacts else Decision.RECONCILE_MODEL_RESULT
    if attempt.state == "evaluation_started":
        return Decision.BLOCK_AMBIGUOUS_EVALUATOR_ATTEMPT
    if attempt.state == "evaluation_terminal":
        return (Decision.RECONCILE_EVALUATION_RESULT if attempt.evaluation_status == "success"
                else Decision.RECONCILE_FAILED_EVALUATION)
    return Decision.BLOCK_CORRUPT_STATE


@dataclass(frozen=True)
class Publication:
    sequence: int
    sha256: str
    created: bool


class Ledger:
    def __init__(self, run_dir, run_id):
        _id(run_id)
        with _io_errors():
            self.root = _checked_path(run_dir)
        self.directory = self.root / DIRECTORY
        self.run_id = run_id

    @classmethod
    def create(cls, run_dir, run_id):
        """Explicit bootstrap of a NEW run; never infer/import old results."""
        ledger = cls(run_dir, run_id)
        with _io_errors():
            if ledger.directory.exists() or ledger.directory.is_symlink():
                ledger._manifest(writing=True)
                return ledger
            _require(not ledger.root.exists() or not any(ledger.root.iterdir()), "LEDGER_LEGACY_READ_ONLY")
            _mkdir(ledger.root)
            ledger.directory.mkdir(mode=0o700)  # A racing bootstrap must not share ownership.
            _sync_dir(ledger.root)
            _mkdir(ledger.directory / "evaluations")
            _publish(ledger.directory / "run.json", {"schema_version": SCHEMA, "run_id": run_id})
        return ledger

    def _manifest(self, *, writing=False):
        data = _read_json(self.directory / "run.json")
        _keys(data, ("schema_version", "run_id"))
        _schema(data["schema_version"])
        _require(data["run_id"] == self.run_id, "LEDGER_IDENTITY_MISMATCH")
        _require(_checked_path(self.directory / "evaluations").is_dir(), "LEDGER_FILE_MISSING")
        _require(not writing or data["schema_version"] == SCHEMA, "LEDGER_SCHEMA_READ_ONLY")
        return data["schema_version"]

    def _folder(self, evaluation_id):
        _id(evaluation_id)
        return _checked_path(self.directory / "evaluations" / evaluation_id)

    def _rows(self, evaluation_id, *, schema_version=SCHEMA):
        folder = self._folder(evaluation_id)
        events = _checked_path(folder / "events")
        head_path = _checked_path(folder / "durable_head.json")
        # Read the watermark first: a concurrent append may extend this prefix,
        # but cannot make the watermark point beyond a later directory listing.
        head = _read_json(head_path) if head_path.exists() else None
        if not events.exists():
            _require(not folder.exists(), "LEDGER_NOT_PLANNED")
            return []
        paths = sorted(p for p in events.iterdir() if not p.name.startswith(".tmp-"))
        rows = []
        for number, path in enumerate(paths, 1):
            _require(path.name == f"{number:012d}.json", "LEDGER_SEQUENCE")
            row = _read_json(path)
            _require(type(row) is dict, "LEDGER_FIELDS_INVALID")
            rows.append(row)
        if head is not None:
            _keys(head, ("schema_version", "sequence", "event_digest"))
            _schema(head["schema_version"], schema_version)
            _require(_integer(head["sequence"], 1) and head["sequence"] <= len(rows), "LEDGER_HEAD_MISSING_EVENT")
            _require(rows[head["sequence"] - 1].get("sha256") == head["event_digest"], "LEDGER_HEAD_MISMATCH")
        return rows

    def _verify_artifacts(self, folder, snapshot, *, sync=False):
        for attempt in snapshot.attempts:
            for artifact in (*attempt.preparation_artifacts, *attempt.model_artifacts, *attempt.result_artifacts,
                             *attempt.authorization_artifacts):
                path = _regular(folder / artifact.path)
                with path.open("rb") as stream:
                    sha = hashlib.sha256()
                    size = 0
                    for data in iter(lambda: stream.read(1024 * 1024), b""):
                        sha.update(data)
                        size += len(data)
                    _require((sha.hexdigest(), size) == (artifact.sha256, artifact.size), "LEDGER_ARTIFACT_DIGEST")
                    if sync:
                        os.fsync(stream.fileno())
                if sync:
                    # Durably bind all directory entries up to the evaluation.
                    for parent in (path.parent, path.parent.parent, folder):
                        _sync_dir(parent)

    def _read(self, evaluation_id, expected=None):
        schema_version = self._manifest()
        _require(expected is None or (isinstance(expected, Identity) and expected.run_id == self.run_id
                 and expected.evaluation_id == evaluation_id), "LEDGER_IDENTITY_MISMATCH")
        rows = self._rows(evaluation_id, schema_version=schema_version)
        if not rows:
            return None
        snapshot = reduce_events(rows, schema_version=schema_version)
        _require(snapshot.identity.run_id == self.run_id and snapshot.identity.evaluation_id == evaluation_id,
                 "LEDGER_IDENTITY_MISMATCH")
        _require(expected is None or snapshot.identity == expected, "LEDGER_IDENTITY_MISMATCH")
        self._verify_artifacts(self._folder(evaluation_id), snapshot)
        return snapshot

    def inspect(self, evaluation_id, expected=None):
        """Read only, including for legacy runs, corruption and abandoned temp files."""
        try:
            with _io_errors():
                _id(evaluation_id)
                _checked_path(self.directory)
                _require(self.root.is_dir(), "LEDGER_RUN_MISSING")
                if not self.directory.exists():
                    return Inspection(legacy=True)
                version = self._manifest()
                return Inspection(snapshot=self._read(evaluation_id, expected), read_only=version == 1)
        except LedgerError as error:
            return Inspection(error_code=error.code)

    def plan(self, identity, *, fault=lambda point: None):
        _require(isinstance(identity, Identity) and identity.run_id == self.run_id, "LEDGER_IDENTITY_MISMATCH")
        with _io_errors():
            self._manifest(writing=True)
            _require(identity.schema_version == SCHEMA, "LEDGER_SCHEMA_READ_ONLY")
            # Only creation/ordinal uniqueness uses a short run lock. No execution
            # or algorithm decision is serialized by this catalogue lock.
            with _exclusive(self.directory / ".catalogue.lock"):
                for path in (self.directory / "evaluations").iterdir():
                    other = self._read(path.name)
                    if other and other.identity.evaluation_id != identity.evaluation_id:
                        _require(other.identity.ordinal != identity.ordinal, "LEDGER_ORDINAL_REUSED")
                folder = self._folder(identity.evaluation_id)
                _mkdir(folder)
                _mkdir(folder / "events")
                _mkdir(folder / "artifacts")
                with _exclusive(folder / ".lock"):
                    session = _Session(self, identity, fault)
                    return session._append("planned", None, {"identity": asdict(identity)})

    def evaluations(self):
        """Read-only inventory ordered by ordinal, never keyed/deduplicated by task."""
        with _io_errors():
            self._manifest()
            snapshots = [self._read(p.name) for p in (self.directory / "evaluations").iterdir()]
            snapshots = [s for s in snapshots if s is not None]
            _require(len({s.identity.ordinal for s in snapshots}) == len(snapshots), "LEDGER_ORDINAL_REUSED")
            return tuple(sorted(snapshots, key=lambda s: s.identity.ordinal))

    @contextmanager
    def session(self, identity, *, fault=lambda point: None):
        """The owner must keep this context open across external operations."""
        with _io_errors():
            self._manifest(writing=True)
            _require(identity.schema_version == SCHEMA, "LEDGER_SCHEMA_READ_ONLY")
            _require(identity.run_id == self.run_id, "LEDGER_IDENTITY_MISMATCH")
            folder = self._folder(identity.evaluation_id)
            _require(folder.is_dir(), "LEDGER_NOT_PLANNED")
            with _exclusive(folder / ".lock"):
                _require(self._read(identity.evaluation_id, identity) is not None, "LEDGER_NOT_PLANNED")
                # A prior owner may have crashed after rename but before fsync.
                # Stabilize the visible validated prefix before any new action;
                # this adds no events and does not rewrite a bound evaluation.
                _sync_dir(folder / "events")
                _sync_dir(folder)
                session = _Session(self, identity, fault)
                try:
                    yield session
                finally:
                    session.active = False


class _Session:
    def __init__(self, ledger, identity, fault):
        self.ledger, self.identity, self.fault = ledger, identity, fault
        self.active = True

    def snapshot(self):
        _require(self.active, "LEDGER_SESSION_CLOSED")
        with _io_errors():
            return self.ledger._read(self.identity.evaluation_id, self.identity)

    def _append(self, kind, attempt_id, data):
        _require(self.active, "LEDGER_SESSION_CLOSED")
        try:
            with _io_errors():
                ledger, identity = self.ledger, self.identity
                ledger._manifest(writing=True)
                current = ledger._read(identity.evaluation_id, identity)
                for event in current.events if current else ():
                    if (event.kind, event.attempt_id) == (kind, attempt_id):
                        _require(event.data == canonical(data), "LEDGER_IDEMPOTENCY_CONFLICT")
                        return Publication(event.sequence, event.sha256, False)
                rows = ledger._rows(identity.evaluation_id)
                self.fault("transition.before")
                body = {"schema_version": SCHEMA, "sequence": len(rows) + 1,
                        "previous_digest": rows[-1]["sha256"] if rows else None,
                        "created_at_ns": time.time_ns(), "kind": kind, "attempt_id": attempt_id,
                        "identity_sha256": digest(asdict(identity)), "data": data}
                event = {**body, "sha256": digest(body)}
                snapshot = reduce_events([*rows, event])
                folder = ledger._folder(identity.evaluation_id)
                ledger._verify_artifacts(folder, snapshot, sync=True)
                _publish(folder / "events" / f'{event["sequence"]:012d}.json', event, fault=self.fault)
                # A replaceable integrity watermark detects missing published
                # tails. It is not an event, a binding or a second state machine.
                _publish(folder / "durable_head.json", {"schema_version": SCHEMA,
                         "sequence": event["sequence"], "event_digest": event["sha256"]},
                         replace_head=True, fault=self.fault, prefix="head")
                self.fault("transition.after")
                return Publication(event["sequence"], event["sha256"], True)
        except BaseException:
            self.active = False  # Never continue past a failed persistence boundary.
            raise

    def reserve(self, attempt_id):
        return self._append("reserved", attempt_id, {})

    def start_preparation(self, attempt_id):
        return self._append("preparation_started", attempt_id, {})

    def finish_preparation(self, attempt_id, *, failure=None, artifacts=None):
        return self._append("preparation_terminal", attempt_id,
                            {"status": "failed" if failure else "success", "error": asdict(failure) if failure else None,
                             **({"artifacts": [asdict(a) for a in artifacts]} if artifacts is not None else {})})

    def start_model(self, attempt_id):
        return self._append("model_started", attempt_id,
                            {"call_id": invocation_id(self.identity, attempt_id, "model")})

    def finish_model(self, attempt_id, *, failure=None, artifacts=(), evidence=None):
        return self._append("model_terminal", attempt_id,
                            {"call_id": invocation_id(self.identity, attempt_id, "model"),
                             "status": "failed" if failure else "success", "error": asdict(failure) if failure else None,
                             "artifacts": [asdict(a) for a in artifacts],
                             **({"evidence": evidence} if evidence is not None else {})})

    def seal_model_result(self, attempt_id, artifacts):
        return self._append("model_result_ready", attempt_id, {"artifacts": [asdict(a) for a in artifacts]})

    def start_evaluation(self, attempt_id):
        return self._append("evaluation_started", attempt_id,
                            {"evaluator_call_id": invocation_id(self.identity, attempt_id, "evaluator")})

    def finish_evaluation(self, attempt_id, artifacts=(), *, failure=None, evidence=None):
        return self._append("evaluation_terminal", attempt_id,
                            {"evaluator_call_id": invocation_id(self.identity, attempt_id, "evaluator"),
                             "status": "failed" if failure else "success", "error": asdict(failure) if failure else None,
                             "artifacts": [asdict(a) for a in artifacts],
                             **({"evidence": evidence} if evidence is not None else {})})

    def confirm_started(self, attempt_id, kind):
        _require(kind in ("model", "evaluation"), "LEDGER_EVENT_INVALID")
        return self._append(kind + "_start_confirmed", attempt_id, {})

    def finish_cleanup(self, attempt_id, *, failure=None):
        return self._append("cleanup_terminal", attempt_id,
                            {"status": "failed" if failure else "success", "error": asdict(failure) if failure else None})

    def bind(self, attempt_id, *, outcome=None):
        """Close a known terminal; optional outcome is an assertion, never an override."""
        current = self.snapshot()
        _require(bool(current.attempts) and current.attempts[-1].attempt_id == attempt_id, "LEDGER_ATTEMPT_INVALID")
        data = _binding(self.identity, current.attempts[-1])
        _require(outcome is None or outcome == data["outcome"], "LEDGER_BINDING_CONFLICT")
        return self._append("bound", attempt_id, data)
