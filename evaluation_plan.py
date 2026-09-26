"""Immutable evaluation identities and a durable, ordered planning journal."""
from dataclasses import asdict, dataclass, replace
import os
from pathlib import Path
import tempfile

import evaluation_ledger as el
from execution_policy import VALIDATED

PLAN_SCHEMA = 1
SCHEDULER = "validated_scheduler_v1"


def check(condition, code):
    if not condition:
        raise el.LedgerError(code)


def durable_json(path, value, *, replace_existing=False, fault=lambda point: None):
    """Run-lock-owned publication; checkpoint replacement has one commit point."""
    path = el._checked_path(path)
    data = el.canonical(value) + b"\n"
    fault("checkpoint.before_write")
    fd, temporary = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    fault("checkpoint.before_publish")
    if replace_existing:
        os.replace(temporary, path)
    else:
        os.link(temporary, path)  # Never overwrite even if an unexpected writer races.
        os.unlink(temporary)
    el._sync_dir(path.parent)
    fault("checkpoint.after_publish")


def model_fingerprint(config):
    return el.digest(asdict(replace(config.codex, sandbox="workspace-write", approve_for_me=True,
        ephemeral=True, output_schema=None, timeout_seconds=config.execution.evaluation_timeout)))


def execution_identity(config, manifest):
    from run_metadata import file_hash, ROOT
    tasks = manifest["tasks"]
    return {"ledger_schema": 2, "plan_schema": PLAN_SCHEMA, "adapter_schema": 1,
            "run_id": manifest["run_id"], "scheduler_identity": SCHEDULER, "ledger_enabled": True,
            "task_membership_hash": tasks["task_membership_hash"], "task_order_hash": tasks["task_order_hash"],
            "source_fingerprint": manifest["policy"]["initial_source_snapshot"]["sha256"],
            "policy_fingerprint": manifest["policy"]["downstream_instructions_sha256"],
            "model_fingerprint": model_fingerprint(config),
            "evaluator_fingerprint": evaluator_fingerprint(config.evaluation.polyglot)}


def evaluator_fingerprint(polyglot=False):
    """Validated evaluator identity: the SWE harness file, or the Polyglot lifecycle set."""
    from run_metadata import file_hash, ROOT
    if polyglot:
        from polyglot_lifecycle import evaluator_fingerprint as polyglot_fingerprint
        return polyglot_fingerprint()
    return file_hash(ROOT / "swe_bench/harness.py")


@dataclass(frozen=True)
class EvaluationPlan:
    identity: el.Identity
    base_commit: str
    run_fingerprint: str
    task_schedule_identity: str
    selection_digest: str | None = None
    search_identity: str | None = None
    schema_version: int = PLAN_SCHEMA

    def __post_init__(self):
        check(type(self.schema_version) is int and self.schema_version == PLAN_SCHEMA, "PLAN_SCHEMA")
        check(self.identity.schema_version == 2, "PLAN_LEDGER_SCHEMA")
        el._id(self.base_commit)
        for value in (self.run_fingerprint, self.task_schedule_identity):
            el._sha(value)
        for value in (self.selection_digest, self.search_identity):
            if value is not None:
                el._sha(value)
        check(self.identity.evaluation_id == self.stable_id(self.body()), "PLAN_IDENTITY")
        check(self.identity.stage != "final_benchmark" or self.selection_digest is not None, "PLAN_SELECTION_REQUIRED")

    def body(self):
        value = asdict(self)
        value["identity"].pop("evaluation_id")
        return value

    @staticmethod
    def stable_id(body):
        return "eval-" + el.digest(body)

    @classmethod
    def create(cls, *, run_id, stage, candidate_id, entry, ordinal, task_draw_index,
               run_fingerprint, task_schedule_identity, model_fingerprint, policy_fingerprint,
               source_fingerprint, evaluator_fingerprint, selection_digest=None, search_identity=None):
        row = dict(identity=asdict(el.Identity(run_id, "placeholder", stage, entry["instance_id"], task_draw_index,
            candidate_id, el.digest(entry), model_fingerprint, policy_fingerprint, source_fingerprint,
            evaluator_fingerprint, ordinal)), base_commit=entry["base_commit"],
            run_fingerprint=run_fingerprint, task_schedule_identity=task_schedule_identity,
            selection_digest=selection_digest, search_identity=search_identity, schema_version=PLAN_SCHEMA)
        body = {**row, "identity": dict(row["identity"])}
        body["identity"].pop("evaluation_id")
        row["identity"]["evaluation_id"] = cls.stable_id(body)
        return cls.from_dict(row)

    @classmethod
    def from_dict(cls, row):
        el._keys(row, cls.__dataclass_fields__)
        return cls(**{**row, "identity": el._identity(row["identity"])})

    def spec(self, attempt_id="attempt-000001"):
        from evaluation_adapter import EvaluationExecutionSpec
        return EvaluationExecutionSpec(self.identity, attempt_id, self.base_commit)


class PlanStore:
    """Caller holds the run lock. Plans never mutate or derive RNG values."""
    def __init__(self, directory, manifest):
        self.root, self.manifest = el._checked_path(directory), manifest
        self.folder = self.root / "evaluation_plans"

    def validate(self, plan):
        m = self.manifest
        check(m["execution_policy"]["effective_policy"] == VALIDATED, "PLAN_POLICY")
        check(plan.identity.run_id == m["run_id"] and plan.run_fingerprint == m["reproducibility_fingerprint"],
              "PLAN_RUN_MISMATCH")
        check(plan.task_schedule_identity == el.digest(m["tasks"]), "PLAN_SCHEDULE_MISMATCH")
        check(plan.identity.task_id in m["tasks"]["ordered_ids"], "PLAN_TASK_MISMATCH")
        check(plan.search_identity == (el.digest(m["experiment"]) if plan.identity.stage == "search" else None),
              "PLAN_SEARCH_IDENTITY")
        for name in ("model_fingerprint", "evaluator_fingerprint"):
            check(getattr(plan.identity, name) == m["evaluation_execution"][name], "PLAN_FINGERPRINT_MISMATCH")

    def read(self):
        if not self.folder.exists():
            return ()
        el._checked_path(self.folder)
        head_path = self.folder / "head.json"
        head = el._read_json(head_path) if head_path.exists() else None
        paths = sorted(p for p in self.folder.iterdir() if not p.name.startswith(".tmp-") and p.name != "head.json")
        records, previous = [], None
        for ordinal, path in enumerate(paths):
            check(path.name == f"{ordinal:012d}.json", "PLAN_SEQUENCE")
            row = el._read_json(path)
            el._keys(row, ("plan", "previous_digest", "sha256"))
            check(row["previous_digest"] == previous and row["sha256"] == el.digest(
                {k: v for k, v in row.items() if k != "sha256"}), "PLAN_CHAIN")
            plan = EvaluationPlan.from_dict(row["plan"])
            self.validate(plan)
            check(plan.identity.ordinal == ordinal, "PLAN_SEQUENCE")
            records.append(plan)
            previous = row["sha256"]
        if head is not None:
            el._keys(head, ("count", "sha256"))
            check(type(head["count"]) is int and 1 <= head["count"] <= len(paths), "PLAN_HEAD")
            check(el._read_json(paths[head["count"] - 1])["sha256"] == head["sha256"], "PLAN_HEAD")
        return tuple(records)

    def publish(self, plan):
        self.validate(plan)
        records = self.read()
        n = plan.identity.ordinal
        if n < len(records):
            check(records[n] == plan, "PLAN_CONFLICT")
            return False
        check(n == len(records), "PLAN_SEQUENCE")
        el._mkdir(self.folder)
        previous = el._read_json(self.folder / f"{n-1:012d}.json")["sha256"] if n else None
        body = {"plan": asdict(plan), "previous_digest": previous}
        row = {**body, "sha256": el.digest(body)}
        durable_json(self.folder / f"{n:012d}.json", row)
        durable_json(self.folder / "head.json", {"count": n+1, "sha256": row["sha256"]}, replace_existing=True)
        return True
