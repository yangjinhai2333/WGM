"""Opt-in single-evaluation execution boundary; no task selection or accounting policy."""
from collections import Counter
from dataclasses import asdict, dataclass, field
import hashlib
import os
import re
import tempfile

import evaluation_ledger as el
from network_readiness import enabled as readiness_enabled, ReadinessAbort

ARTIFACT_NAMES = frozenset(("preparation.json", "patch.diff", "prediction.json", "grader.json",
                            "task-metadata.json", "test-output.safe.json", "preparation-error.safe.json",
                            "readiness-error.safe.json"))
MAX_PATCH_BYTES = 1024 * 1024


class AdapterError(ValueError):
    """Only fixed codes may leave this module."""

    def __init__(self, code):
        self.code = code
        super().__init__(code)


def require(value, code="ADAPTER_INVALID_RESULT"):
    if not value:
        raise AdapterError(code)


@dataclass(frozen=True)
class EvaluationExecutionSpec:
    identity: el.Identity
    attempt_id: str
    base_commit: str

    def __post_init__(self):
        require(isinstance(self.identity, el.Identity), "ADAPTER_IDENTITY_MISMATCH")
        el._id(self.attempt_id)
        el._id(self.base_commit)

    @property
    def model_call_id(self):
        return el.invocation_id(self.identity, self.attempt_id, "model")

    @property
    def evaluator_call_id(self):
        return el.invocation_id(self.identity, self.attempt_id, "evaluator")

    def envelope(self):
        return {"adapter_schema":1, **asdict(self), "model_call_id": self.model_call_id,
                "evaluator_call_id": self.evaluator_call_id}

    def preparation_envelope(self):
        # Preparation needs execution identity, never prospective invocation IDs.
        return {"adapter_schema": 1, "preparation_context_schema": 2, **asdict(self)}


def validate_preparation_context(spec, context):
    expected = (spec.preparation_envelope() if isinstance(context, dict)
                and context.get("preparation_context_schema") == 2 else spec.envelope())
    require(el.same_json(context, expected), "ADAPTER_IDENTITY_MISMATCH")


@dataclass(frozen=True)
class ExecutionEvidence:
    started: bool
    reaped: bool
    returncode: int | None
    launch_failed: bool = False
    timed_out: bool = False
    interrupted: bool = False
    duration_seconds: float = 0.0
    input_tokens: int | None = None
    cached_input_tokens: int | None = None
    output_tokens: int | None = None
    usage_error_code: str | None = None

    @property
    def terminal(self):
        try:
            el.validate_execution_evidence(asdict(self))
            return True
        except el.LedgerError:
            return False

    @property
    def error_code(self):
        if self.launch_failed:
            return "MODEL_FAILED"
        if self.timed_out:
            return "TIMEOUT"
        if self.interrupted:
            return "INTERRUPTED"
        return None


@dataclass(frozen=True)
class PreparedEvaluation:
    spec: EvaluationExecutionSpec
    workspace_fingerprint: str
    source_fingerprint: str
    testspec_fingerprint: str
    image_fingerprint: str
    baseline: str
    git_config_sha256: str
    resource_id: str

    def __post_init__(self):
        for name in ("workspace_fingerprint", "source_fingerprint", "testspec_fingerprint", "image_fingerprint", "git_config_sha256"):
            el._sha(getattr(self, name))
        el._id(self.baseline)
        el._id(self.resource_id)
        require(self.source_fingerprint == self.spec.identity.source_fingerprint, "ADAPTER_IDENTITY_MISMATCH")


@dataclass(frozen=True)
class ModelExecutionResult:
    spec: EvaluationExecutionSpec
    call_id: str
    evidence: ExecutionEvidence
    error_code: str | None = None


@dataclass(frozen=True)
class SealedModelArtifacts:
    spec: EvaluationExecutionSpec
    patch: bytes = field(repr=False)

    @property
    def patch_sha256(self):
        return hashlib.sha256(self.patch).hexdigest()


@dataclass(frozen=True)
class EvaluatorExecutionResult:
    spec: EvaluationExecutionSpec
    call_id: str
    evidence: ExecutionEvidence
    # Canonical, immutable JSON of the actual grader result. Never raw stdout.
    report: bytes = field(repr=False)
    error_code: str | None = None


@dataclass(frozen=True)
class ValidatedEvaluationResult:
    spec: EvaluationExecutionSpec
    resolved: bool
    empty_patch: bool
    patch_sha256: str
    artifacts: tuple[el.Artifact, ...]


@dataclass(frozen=True)
class EvaluationExecutionOutcome:
    spec: EvaluationExecutionSpec
    decision: str
    outcome: str | None = None
    binding_id: str | None = None
    resolved: bool | None = None
    error_code: str | None = None
    cleanup_status: str | None = None


def _identity(spec, value):
    require(value == spec, "ADAPTER_IDENTITY_MISMATCH")


def _error(evidence, explicit, *, model):
    if explicit is not None:
        el.Failure(explicit)
    code = evidence.error_code or explicit
    if model and evidence.returncode != 0:
        code = code or "MODEL_FAILED"
    if not model and evidence.launch_failed:
        code = "EVALUATOR_FAILED"
    return el.Failure(code) if code else None


class EvaluationLifecycleAdapter:
    """Callbacks are trusted boundary implementations; invocation rights stay here.

    prepare(spec, workdir), recover_preparation(spec, workdir),
    restore(prepared, workdir, for_model=False), model(prepared, workdir, on_started),
    seal_model(prepared, workdir), evaluate(prepared, sealed, workdir, on_started),
    cleanup(spec, workdir, prepared=None). No callback is called by SKIP_BOUND.
    """

    def __init__(self, ledger, callbacks, *, fault=lambda point: None):
        self.ledger, self.callbacks, self.fault = ledger, callbacks, fault

    def _folder(self, spec):
        return self.ledger._folder(spec.identity.evaluation_id)

    def _workdir(self, spec):
        return el._checked_path(self._folder(spec) / "work" / spec.attempt_id)

    def _store(self, spec, name, data):
        """Immutable safe artifacts, durably published before a referencing event."""
        el._id(name)
        require(name in ARTIFACT_NAMES and type(data) is bytes, "ADAPTER_ARTIFACT_INVALID")
        require(len(data) <= (MAX_PATCH_BYTES if name == "patch.diff" else el.MAX_JSON_BYTES),
                "ADAPTER_ARTIFACT_SIZE")
        folder = self._folder(spec) / "artifacts" / spec.attempt_id
        el._mkdir(folder)
        path = el._checked_path(folder / name)
        if path.exists():
            require(el._regular(path).read_bytes() == data, "ADAPTER_ARTIFACT_CONFLICT")
        else:
            fd, temporary = tempfile.mkstemp(prefix=".tmp-", dir=folder)
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.rename(temporary, path)
            el._sync_dir(folder)
        return el.Artifact(f"artifacts/{spec.attempt_id}/{name}", hashlib.sha256(data).hexdigest(), len(data))

    def _json(self, spec, name, value):
        return self._store(spec, name, el.canonical(value))

    def _load(self, spec, name):
        require(name in ARTIFACT_NAMES and name.endswith(".json"), "ADAPTER_ARTIFACT_INVALID")
        result = el._read_json(self._folder(spec) / "artifacts" / spec.attempt_id / name)
        require(type(result) is dict, "ADAPTER_BUSINESS_MISMATCH")
        return result

    def _prepared(self, spec):
        row = self._load(spec, "preparation.json")
        validate_preparation_context(spec, row.pop("context", None))
        return PreparedEvaluation(spec, **row)

    def _finish_preparation(self, session, spec, prepared):
        require(isinstance(prepared, PreparedEvaluation))
        _identity(spec, prepared.spec)
        row = asdict(prepared)
        row.pop("spec")
        context = (spec.preparation_envelope() if readiness_enabled(getattr(self.callbacks, "config", None))
                   else spec.envelope())
        artifact = self._json(spec, "preparation.json", {"context": context, **row})
        session.finish_preparation(spec.attempt_id, artifacts=[artifact])

    def _seal(self, session, spec, prepared, workdir):
        self.fault("seal.before")
        artifact_dir = self._folder(spec) / "artifacts" / spec.attempt_id
        if (artifact_dir / "patch.diff").exists() and (artifact_dir / "prediction.json").exists():
            sealed = self._sealed(spec)  # A crash may precede the seal event, not artifact publication.
        else:
            sealed = self.callbacks.seal_model(prepared, workdir)
        require(isinstance(sealed, SealedModelArtifacts) and type(sealed.patch) is bytes)
        _identity(spec, sealed.spec)
        sealed.patch.decode("utf-8")  # Only textual task patches are supported.
        patch = self._store(spec, "patch.diff", sealed.patch)
        prediction = self._json(spec, "prediction.json", {"context": spec.envelope(),
            "instance_id": spec.identity.task_id, "patch_sha256": sealed.patch_sha256,
            "empty_patch": not bool(sealed.patch)})
        self.fault("seal.after_artifacts")
        session.seal_model_result(spec.attempt_id, sorted((patch, prediction), key=lambda a:a.path))
        self.fault("seal.after")
        return sealed

    def _sealed(self, spec):
        prediction = self._load(spec, "prediction.json")
        path = el._regular(self._folder(spec) / "artifacts" / spec.attempt_id / "patch.diff")
        require(path.stat().st_size <= MAX_PATCH_BYTES, "ADAPTER_ARTIFACT_SIZE")
        patch = path.read_bytes()
        expected = {"context":spec.envelope(), "instance_id":spec.identity.task_id,
                    "patch_sha256":hashlib.sha256(patch).hexdigest(), "empty_patch":not bool(patch)}
        require(el.same_json(prediction, expected), "ADAPTER_BUSINESS_MISMATCH")
        return SealedModelArtifacts(spec, patch)

    def _result_artifacts(self, spec, sealed, result):
        # Keep only typed, whitelisted grader fields, including their actual ID.
        report = el._decode(result.report)
        require(type(report) is dict and set(report) == {"instance_id", "resolved", "status", "error"},
                "ADAPTER_BUSINESS_MISMATCH")
        require(type(report["instance_id"]) is str and type(report["resolved"]) is bool
                and report["status"] == "completed" and report["error"] is None, "ADAPTER_BUSINESS_MISMATCH")
        require(report["instance_id"] == spec.identity.task_id, "ADAPTER_BUSINESS_MISMATCH")
        common = {"context": spec.envelope(), "patch_sha256": sealed.patch_sha256}
        metadata = {**common, "instance_id":spec.identity.task_id, "status":"completed",
                    "resolved":report["resolved"], "error":None, "usage":asdict(result.evidence)}
        artifacts = [self._json(spec, "grader.json", {**common, **report}),
                     self._json(spec, "task-metadata.json", metadata),
                     self._json(spec, "test-output.safe.json", {**common,
                         "status":"completed", "resolved":report["resolved"], "raw_output_omitted":True})]
        return tuple(sorted(artifacts, key=lambda a:a.path))

    def _validate(self, spec, snapshot):
        attempt = snapshot.attempts[-1]
        for artifacts, names in ((attempt.preparation_artifacts, {"preparation.json"}),
                                 (attempt.model_artifacts, {"patch.diff", "prediction.json"}),
                                 (attempt.result_artifacts, {"grader.json", "task-metadata.json", "test-output.safe.json"})):
            require({a.path for a in artifacts} == {f"artifacts/{spec.attempt_id}/{n}" for n in names},
                    "ADAPTER_ARTIFACT_INVALID")
        sealed = self._sealed(spec)
        common = {"context":spec.envelope(), "patch_sha256":sealed.patch_sha256}
        report, metadata = self._load(spec, "grader.json"), self._load(spec, "task-metadata.json")
        safe = self._load(spec, "test-output.safe.json")
        resolved = report.get("resolved")
        require(type(resolved) is bool, "ADAPTER_BUSINESS_MISMATCH")
        require(el.same_json(report, {**common, "instance_id":spec.identity.task_id,
                          "resolved":resolved, "status":"completed", "error":None}), "ADAPTER_BUSINESS_MISMATCH")
        require(set(metadata) == {*common, "instance_id", "resolved", "status", "error", "usage"},
                "ADAPTER_BUSINESS_MISMATCH")
        el.validate_execution_evidence(metadata["usage"])
        require(el.same_json(metadata, {**common, "instance_id":spec.identity.task_id, "resolved":resolved,
                            "status":"completed", "error":None, "usage":metadata["usage"]}), "ADAPTER_BUSINESS_MISMATCH")
        terminal = next(e for e in reversed(snapshot.events) if e.kind == "evaluation_terminal")
        require(el.same_json(el._decode(terminal.data).get("evidence"), metadata["usage"]), "ADAPTER_BUSINESS_MISMATCH")
        require(el.same_json(safe, {**common, "status":"completed", "resolved":resolved, "raw_output_omitted":True}),
                "ADAPTER_BUSINESS_MISMATCH")
        return ValidatedEvaluationResult(spec, resolved, not bool(sealed.patch), sealed.patch_sha256,
                                         snapshot.attempts[-1].result_artifacts)

    def _outcome(self, spec, session, *, decision=None, error=None, resolved=None):
        snapshot = session.snapshot()
        binding = next((el._decode(e.data) for e in reversed(snapshot.events) if e.kind == "bound"), None)
        return EvaluationExecutionOutcome(spec, decision or el.classify(el.Inspection(snapshot)).value,
            binding["outcome"] if binding else None, snapshot.binding_id, resolved,
            error or (binding["error_code"] if binding else None),
            snapshot.attempts[-1].cleanup_status if snapshot.attempts else None)

    def _cleanup(self, spec, session, workdir):
        self.fault("cleanup.before")
        failure = None
        try:
            path = self._folder(spec) / "artifacts" / spec.attempt_id / "preparation.json"
            prepared = self._prepared(spec) if path.exists() else None
            self.callbacks.cleanup(spec, workdir, prepared=prepared)
        except (Exception, KeyboardInterrupt):
            failure = el.Failure("CLEANUP_FAILED")
        session.finish_cleanup(spec.attempt_id, failure=failure)
        self.fault("cleanup.after")

    def _record_error(self, spec, error):
        """Diagnostic only: never a terminal, invocation permission or retry grant."""
        name = type(error).__name__
        name = name if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]{0,63}', name) else 'UnknownException'
        codes = {'ADAPTER_IDENTITY_MISMATCH', 'ADAPTER_RESOURCE_MISMATCH',
                 'ADAPTER_SOURCE_MISMATCH', 'ADAPTER_WORKSPACE_INVALID',
                 'ADAPTER_TESTSPEC_MISMATCH', 'ADAPTER_IMAGE_MISMATCH',
                 'ADAPTER_PREPARATION_UNKNOWN', 'ADAPTER_BUSINESS_MISMATCH',
                 'ADAPTER_READINESS_STATE_CHANGED', 'ADAPTER_READINESS_GATE_REQUIRED'}
        code = getattr(error, 'code', None)
        code = code if isinstance(error, AdapterError) and code in codes else 'ADAPTER_ARTIFACT_OR_STATE_INVALID'
        inspection = self.ledger.inspect(spec.identity.evaluation_id, spec.identity)
        snapshot = inspection.snapshot
        row = {'schema_version': 1, 'context': spec.preparation_envelope(),
               'stage': getattr(self, '_stage', 'adapter_state'), 'exception_type': name,
               'error_code': code, 'ledger_error_code': inspection.error_code,
               'ledger_head': snapshot.events[-1].sha256 if snapshot else None,
               'diagnostic_only': True}
        # Bounded, immutable, outside business artifacts and the event chain.
        folder = self._folder(spec) / 'adapter_errors'
        el._mkdir(folder)
        path = folder / (el.digest(row) + '.json')
        if not path.exists():
            el._publish(path, row)

    def execute(self, spec):
        require(isinstance(spec, EvaluationExecutionSpec), "ADAPTER_IDENTITY_MISMATCH")
        inspection = self.ledger.inspect(spec.identity.evaluation_id)
        if inspection.read_only:
            return EvaluationExecutionOutcome(spec, el.Decision.READ_ONLY.value, error_code="LEDGER_SCHEMA_READ_ONLY")
        inspection = self.ledger.inspect(spec.identity.evaluation_id, spec.identity)
        if inspection.error_code or inspection.legacy:
            return EvaluationExecutionOutcome(spec, el.classify(inspection).value, error_code=inspection.error_code)
        if inspection.snapshot is None:
            self.ledger.plan(spec.identity)
        # This read is authoritative. The inspection above never authorizes a call.
        with self.ledger.session(spec.identity) as session:
            snapshot = session.snapshot()
            entry = el.classify(el.Inspection(snapshot))
            if entry == el.Decision.SKIP_BOUND:
                require(snapshot.attempts[-1].attempt_id == spec.attempt_id, "ADAPTER_ATTEMPT_MISMATCH")
                try:
                    self._prepared(spec)  # Check the execution spec as well as ledger Identity.
                    binding = next(el._decode(e.data) for e in reversed(snapshot.events) if e.kind == "bound")
                    resolved = self._validate(spec, snapshot).resolved if binding["outcome"] == "success" else None
                except (AdapterError, el.LedgerError, OSError, TypeError, KeyError):
                    return EvaluationExecutionOutcome(spec, "BLOCK_ADAPTER_STATE", error_code="ADAPTER_IDENTITY_OR_RESULT_INVALID")
                return self._outcome(spec, session, resolved=resolved)  # No callbacks, writes, attempts or IDs.
            if entry.value.startswith("BLOCK_"):
                return self._outcome(spec, session)
            if snapshot.attempts and snapshot.attempts[-1].attempt_id != spec.attempt_id:
                require(entry in (el.Decision.RETRY_PREPARATION, el.Decision.RETRY_AUTHORIZED)
                        and snapshot.attempts[-1].cleanup_status == "success",
                        "ADAPTER_ATTEMPT_MISMATCH")
                if entry == el.Decision.RETRY_AUTHORIZED:
                    from preparation_recovery import execution_spec as authorized_spec
                    from evaluation_plan import PlanStore
                    manifest = el._read_json(self.ledger.root / 'run_manifest.json')
                    plan = next(p for p in PlanStore(self.ledger.root, manifest).read() if p.identity == spec.identity)
                    require(authorized_spec(self.ledger, plan, el.Inspection(snapshot)) == spec,
                            'ADAPTER_ATTEMPT_MISMATCH')
                session.reserve(spec.attempt_id)
                entry = el.Decision.START_PREPARATION
            if entry == el.Decision.START_NEW:
                session.reserve(spec.attempt_id)
                entry = el.Decision.START_PREPARATION
            if entry == el.Decision.RETRY_PREPARATION:
                return self._outcome(spec, session)
            workdir = self._workdir(spec)
            if entry in (el.Decision.START_PREPARATION, el.Decision.RECOVER_PREPARATION):
                if entry == el.Decision.START_PREPARATION:
                    publication = session.start_preparation(spec.attempt_id)
                    require(publication.created, "ADAPTER_NO_NEW_MARKER")
                self.fault("preparation.before")
                try:
                    el._mkdir(workdir.parent)
                    el._mkdir(workdir)
                    method = self.callbacks.prepare if entry == el.Decision.START_PREPARATION else self.callbacks.recover_preparation
                    prepared = method(spec, workdir)
                except (Exception, KeyboardInterrupt) as error:
                    if entry == el.Decision.RECOVER_PREPARATION and (
                            not isinstance(error, AdapterError) or error.code != "ADAPTER_PREPARATION_INCOMPLETE"):
                        return self._outcome(spec, session, decision="BLOCK_PREPARATION_RECOVERY",
                                             error="ADAPTER_PREPARATION_UNKNOWN")
                    # Preserve diagnostic type, never raw exception text (which may contain secrets).
                    artifact = self._json(spec, 'preparation-error.safe.json', {
                        'stage': 'preparation', 'error_code': 'PREPARATION_FAILED',
                        'exception_type': type(error).__name__})
                    session.finish_preparation(spec.attempt_id, failure=el.Failure("PREPARATION_FAILED"),
                                               artifacts=[artifact])
                    self._cleanup(spec, session, workdir)
                    return self._outcome(spec, session, error="PREPARATION_FAILED")
                self.fault("preparation.after")
                self._finish_preparation(session, spec, prepared)
                if entry == el.Decision.RECOVER_PREPARATION:
                    return self._outcome(spec, session)
            try:
                return self._execute_prepared(spec, session, workdir, entry)
            except (Exception, KeyboardInterrupt) as error:
                # Preserve the verified journal. Never replace uncertain work with a terminal.
                # Persistence failures poison sessions, so do not read/write via that owner.
                # SDK/callback failures can contain endpoints or private output.
                self._record_error(spec, error)
                return EvaluationExecutionOutcome(spec, "BLOCK_ADAPTER_STATE", error_code="ADAPTER_ARTIFACT_OR_STATE_INVALID")

    def _execute_prepared(self, spec, session, workdir, entry):
        self._stage = 'read_preparation'
        decision = el.classify(el.Inspection(session.snapshot()))
        prepared = self._prepared(spec)
        if decision in (el.Decision.RECONCILE_FAILED_MODEL, el.Decision.RECONCILE_FAILED_EVALUATION):
            if spec.identity.stage == 'final_benchmark' and decision == el.Decision.RECONCILE_FAILED_MODEL:
                from evaluation_plan import PlanStore
                from final_readiness_recovery import execution_spec
                manifest = el._read_json(self.ledger.root / 'run_manifest.json')
                plan = next(p for p in PlanStore(self.ledger.root, manifest).read() if p.identity == spec.identity)
                execution_spec(self.ledger, plan, el.Inspection(session.snapshot()))
            self.fault("binding.before")
            session.bind(spec.attempt_id, outcome="failed")
            self.fault("binding.after")
            self._cleanup(spec, session, workdir)
            return self._outcome(spec, session)
        if decision == el.Decision.RECONCILE_EVALUATION_RESULT:
            validated = self._validate(spec, session.snapshot())
            self.fault("binding.before")
            session.bind(spec.attempt_id, outcome="success")
            self.fault("binding.after")
            self._cleanup(spec, session, workdir)
            return self._outcome(spec, session, resolved=validated.resolved)
        if decision != el.Decision.RECONCILE_MODEL_RESULT:
            self._stage = 'restore_preparation'
            self.callbacks.restore(prepared, workdir, for_model=decision == el.Decision.START_MODEL)
        if decision == el.Decision.START_MODEL:
            self._stage = 'model_readiness'
            gate = getattr(self.callbacks, "await_model_readiness", None)
            require(gate is not None or not readiness_enabled(getattr(self.callbacks, "config", None)),
                    "ADAPTER_READINESS_GATE_REQUIRED")
            if gate is not None:
                before = session.snapshot()
                while True:
                    gate()
                    after = self.ledger.inspect(spec.identity.evaluation_id, spec.identity)
                    require(not after.error_code and after.snapshot == before
                            and el.classify(after) == el.Decision.START_MODEL, "ADAPTER_READINESS_STATE_CHANGED")
                    confirm = getattr(self.callbacks, "validate_model_readiness", None)
                    if confirm is None or confirm() is not False:
                        break
            self.fault("model.before_marker")
            self._stage = 'model_marker'
            publication = session.start_model(spec.attempt_id)
            self.fault("model.after_marker")
            require(publication.created, "ADAPTER_NO_NEW_MARKER")
            try:
                self._stage = 'model_call'
                result = self.callbacks.model(prepared, workdir,
                    lambda: session.confirm_started(spec.attempt_id, "model"))
            except ReadinessAbort as error:
                result = getattr(error, 'model_result', None)
                if isinstance(result, ModelExecutionResult) and result.evidence.terminal:
                    _identity(spec, result.spec)
                    require(result.call_id == spec.model_call_id, "ADAPTER_CALL_ID_MISMATCH")
                    artifacts, code = (), 'INTERRUPTED'
                    if (spec.identity.stage == 'final_benchmark'
                            and error.code == 'NETWORK_READINESS_RECEIPT_INVALID'
                            and getattr(error, 'not_started_boundary', None) in
                                ('usage.before_runner', 'runner.before_execute')
                            and not result.evidence.started and result.evidence.launch_failed):
                        code = 'READINESS_RECEIPT_REJECTED'
                        artifacts = (self._json(spec, 'readiness-error.safe.json', {
                            'context': spec.envelope(), 'physical_process_started': False,
                            'boundary': error.not_started_boundary,
                            'rejection': error.receipt_rejection}),)
                    session.finish_model(spec.attempt_id, failure=el.Failure(code), artifacts=artifacts,
                                         evidence=asdict(result.evidence))
                raise  # Stop, with a trusted terminal only when the runner supplied one.
            except (Exception, KeyboardInterrupt):
                return self._outcome(spec, session, error="ADAPTER_MODEL_TERMINAL_UNKNOWN")
            self.fault("model.after_call")
            require(isinstance(result, ModelExecutionResult))
            require(isinstance(result.evidence, ExecutionEvidence))
            _identity(spec, result.spec)
            require(result.call_id == spec.model_call_id, "ADAPTER_CALL_ID_MISMATCH")
            if not result.evidence.terminal:
                return self._outcome(spec, session, error="ADAPTER_MODEL_TERMINAL_UNKNOWN")
            failure = _error(result.evidence, result.error_code, model=True)
            session.finish_model(spec.attempt_id, failure=failure, evidence=asdict(result.evidence))
            self.fault("model.after_terminal")
            if failure:
                return self._execute_prepared(spec, session, workdir, entry)
            decision = el.Decision.RECONCILE_MODEL_RESULT
        if decision == el.Decision.RECONCILE_MODEL_RESULT:
            self._stage = 'seal_model'
            sealed = self._seal(session, spec, prepared, workdir)
            if entry == el.Decision.RECONCILE_MODEL_RESULT:
                return self._outcome(spec, session)
        else:
            sealed = self._sealed(spec)
        self.fault("evaluation.before_marker")
        self._stage = 'evaluator_marker'
        publication = session.start_evaluation(spec.attempt_id)
        self.fault("evaluation.after_marker")
        require(publication.created, "ADAPTER_NO_NEW_MARKER")
        try:
            self._stage = 'evaluator_call'
            result = self.callbacks.evaluate(prepared, sealed, workdir,
                lambda: session.confirm_started(spec.attempt_id, "evaluation"))
        except (Exception, KeyboardInterrupt):
            return self._outcome(spec, session, error="ADAPTER_EVALUATOR_TERMINAL_UNKNOWN")
        self.fault("evaluation.after_call")
        require(isinstance(result, EvaluatorExecutionResult))
        require(isinstance(result.evidence, ExecutionEvidence))
        _identity(spec, result.spec)
        require(result.call_id == spec.evaluator_call_id, "ADAPTER_CALL_ID_MISMATCH")
        if not result.evidence.terminal:
            return self._outcome(spec, session, error="ADAPTER_EVALUATOR_TERMINAL_UNKNOWN")
        failure = _error(result.evidence, result.error_code, model=False)
        artifacts = ()
        if not failure:
            try:
                artifacts = self._result_artifacts(spec, sealed, result)
            except (AdapterError, el.LedgerError, TypeError, UnicodeError):
                failure = el.Failure("ARTIFACT_INVALID")
        session.finish_evaluation(spec.attempt_id, artifacts, failure=failure, evidence=asdict(result.evidence))
        self.fault("evaluation.after_terminal")
        return self._execute_prepared(spec, session, workdir, entry)


def execution_inventory(ledger):
    """Read-only observations, not charges, rewards or scheduling decisions."""
    rows = []
    for snapshot in ledger.evaluations():
        identity = snapshot.identity
        attempts = []
        for attempt in snapshot.attempts:
            row = {"attempt_id":attempt.attempt_id, "preparation_status":attempt.preparation_status,
                   "cleanup_status":attempt.cleanup_status,
                   "cleanup_pending":attempt.state == "bound" and attempt.cleanup_status is None,
                   "preparation_started":any(e.kind == "preparation_started" and e.attempt_id == attempt.attempt_id
                                             for e in snapshot.events)}
            for kind, prefix in (("model", "model"), ("evaluation", "evaluator")):
                events = [e for e in snapshot.events if e.attempt_id == attempt.attempt_id]
                terminal = next((el._decode(e.data) for e in events if e.kind == kind+"_terminal"), None)
                marker = next((el._decode(e.data) for e in events if e.kind == kind+"_started"), None)
                evidence = terminal.get("evidence") if terminal else None
                confirmed = getattr(attempt, kind+"_start_confirmed")
                row[prefix] = {"attempted":marker is not None,
                    "started": evidence["started"] if evidence else (True if confirmed else (None if marker else False)),
                    "terminal":terminal["status"] if terminal else None,
                    "ambiguous":marker is not None and terminal is None,
                    "call_id":marker.get("call_id", marker.get("evaluator_call_id")) if marker else None,
                    "evidence":evidence}
            attempts.append(row)
        binding = next((el._decode(e.data) for e in reversed(snapshot.events) if e.kind == "bound"), None)
        rows.append({"identity":asdict(identity), "planned":True, "binding":binding, "attempts":attempts})
    tasks = Counter(r["identity"]["task_id"] for r in rows)
    def counts(selected):
        result = {"planned":len(selected), "bound":sum(r["binding"] is not None for r in selected)}
        for kind in ("model", "evaluator"):
            calls = [a[kind] for r in selected for a in r["attempts"] if a[kind]["attempted"]]
            proof = [c["evidence"] for c in calls if c["evidence"]]
            result[kind] = {"attempted":len(calls), "started":sum(c["started"] is True for c in calls),
                "not_started":sum(c["started"] is False for c in calls),
                "started_unknown":sum(c["started"] is None for c in calls),
                "terminal_success":sum(c["terminal"] == "success" for c in calls),
                "terminal_failed":sum(c["terminal"] == "failed" for c in calls),
                "ambiguous":sum(c["ambiguous"] for c in calls),
                "known_duration_seconds":sum(e["duration_seconds"] for e in proof),
                "duration_complete":len(proof) == len(calls)}
            for field in ("input_tokens", "cached_input_tokens", "output_tokens"):
                values = [e[field] for e in proof]
                result[kind][field] = sum(values) if len(proof) == len(calls) and all(v is not None for v in values) else None
        return result
    groups = {field:{key:counts([r for r in rows if r["identity"][field] == key])
                     for key in sorted({r["identity"][field] for r in rows})}
              for field in ("stage", "candidate_id", "task_id")}
    return {"schema_version":1, "evaluations":rows, "totals":counts(rows), "groups":groups, "planned":len(rows),
            "bound":sum(r["binding"] is not None for r in rows),
            "success_bindings":sum(bool(r["binding"] and r["binding"]["outcome"] == "success") for r in rows),
            "failed_bindings":sum(bool(r["binding"] and r["binding"]["outcome"] == "failed") for r in rows),
            "reserved_attempts":sum(len(r["attempts"]) for r in rows),
            "preparation_attempts":sum(a["preparation_started"] for r in rows for a in r["attempts"]),
            "unique_task_count":len(tasks), "repeated_task_count":sum(n-1 for n in tasks.values()),
            "task_counts":dict(sorted(tasks.items()))}
