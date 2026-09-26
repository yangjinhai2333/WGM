"""Serial validated scheduling. The journal owns execution; SearchState owns consumption."""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import evaluation_ledger as el
from evaluation_adapter import EvaluationLifecycleAdapter
from evaluation_plan import EvaluationPlan, PlanStore, SCHEDULER, check, durable_json
from execution_policy import identity as policy_identity, semantics as policy_semantics


class SchedulingBlocked(RuntimeError):
    def __init__(self, code, plan=None):
        self.code, self.plan = code, plan
        # A plan has no invocation until a real attempt publishes its marker.
        # Read actual call IDs from the ledger, never invent them for an error.
        self.details = ({"evaluation_id": plan.identity.evaluation_id, "task_id": plan.identity.task_id}
                        if plan else {})
        super().__init__(code + (" " + el.canonical(self.details).decode() if self.details else ""))


@dataclass(frozen=True)
class BoundResult:
    plan: EvaluationPlan
    binding_digest: str
    outcome: str
    resolved: bool | None
    empty_patch: bool | None


def read_binding(ledger, plan):
    """Strict read-only business validation. Never constructs a callback or grants execution."""
    inspection = ledger.inspect(plan.identity.evaluation_id, plan.identity)
    check(not inspection.error_code and not inspection.legacy and not inspection.read_only, "BINDING_IDENTITY")
    snapshot = inspection.snapshot
    check(snapshot is not None and snapshot.state == "bound", "BINDING_NOT_READY")
    from preparation_recovery import execution_spec
    spec = execution_spec(ledger, plan, inspection)
    check(snapshot.attempts[-1].attempt_id == spec.attempt_id, "BINDING_ATTEMPT")
    binding = next(el._decode(e.data) for e in reversed(snapshot.events) if e.kind == "bound")
    adapter = EvaluationLifecycleAdapter(ledger, None)
    if binding["outcome"] == "success":
        adapter._prepared(spec)
        value = adapter._validate(spec, snapshot)
        return BoundResult(plan, snapshot.binding_id, "success", value.resolved, value.empty_patch)
    return BoundResult(plan, snapshot.binding_id, "failed", None, None)


def resume_decision(plan, inspection, consumed):
    """Pure classification. Inspection by itself never authorizes external work."""
    decision = el.classify(inspection, plan.identity).value
    if inspection.snapshot is None and not inspection.error_code:
        return "UNPLANNED"
    if decision == "SKIP_BOUND":
        snapshot = inspection.snapshot
        binding = next(el._decode(e.data) for e in reversed(snapshot.events) if e.kind == "bound")
        previous = consumed.get(plan.identity.evaluation_id)
        if previous is not None and previous != snapshot.binding_id:
            return "BLOCK_BINDING_DIGEST_MISMATCH"
        if binding["outcome"] == "failed":
            return "BOUND_FAILURE"
        return "BOUND_CONSUMED" if previous else "BOUND_PENDING_CONSUMPTION"
    return "PLANNED_UNRESERVED" if decision == "START_NEW" else decision


class ExecutionRun:
    def __init__(self, directory, manifest):
        self.directory, self.manifest = el._checked_path(directory), deepcopy(manifest)
        self.ledger = el.Ledger(self.directory, manifest["run_id"])
        self.plans = PlanStore(self.directory, manifest)
        self.verify()

    @classmethod
    def create(cls, directory, manifest):
        directory = el._checked_path(directory)
        # Ledger bootstrap must precede all other files and may only own a fresh directory.
        el.Ledger.create(directory, manifest["run_id"])
        durable_json(directory / "run_manifest.json", manifest)
        return cls(directory, manifest)

    def verify(self):
        from run_metadata import fingerprint_inputs, digest
        saved = el._read_json(self.directory / "run_manifest.json")
        check(el.same_json(saved, self.manifest), "SCHEDULER_MANIFEST_CHANGED")
        check(policy_semantics(saved.get("execution_policy")) == policy_semantics(policy_identity("validated")),
              "SCHEDULER_POLICY_MISMATCH")
        check(saved.get("evaluation_execution", {}).get("scheduler_identity") == SCHEDULER, "SCHEDULER_SCHEMA")
        execution = saved["evaluation_execution"]
        check(all(type(execution.get(k)) is int and execution[k] == v for k,v in
                  (("ledger_schema", 2), ("plan_schema", 1), ("adapter_schema", 1))), "SCHEDULER_SCHEMA")
        check(execution.get("ledger_enabled") is True and all(execution.get(k) == saved["tasks"].get(k)
              for k in ("task_membership_hash", "task_order_hash")), "SCHEDULER_TASK_IDENTITY")
        check(saved["fingerprint_inputs"] == fingerprint_inputs(saved)
              and saved["reproducibility_fingerprint"] == digest(saved["fingerprint_inputs"]), "SCHEDULER_FINGERPRINT")
        check(saved["evaluation_execution"]["run_id"] == saved["run_id"], "SCHEDULER_RUN_ID")
        check(self.ledger._manifest() == 2, "LEDGER_SCHEMA_READ_ONLY")

    @contextmanager
    def lock(self):
        with el._exclusive(self.directory / "evaluation_scheduler.lock"):
            self.verify()
            yield self

    def precheck(self, consumed=None):
        """Verify every persisted identity before authorizing any new evaluation."""
        self.verify()
        plans = self.plans.read()
        by_id = {p.identity.evaluation_id: p for p in plans}
        for snapshot in self.ledger.evaluations():
            check(snapshot.identity.evaluation_id in by_id, "SCHEDULER_UNCLAIMED_EVALUATION")
            check(snapshot.identity == by_id[snapshot.identity.evaluation_id].identity, "SCHEDULER_IDENTITY")
        rows = []
        for p in plans:
            inspection = self.ledger.inspect(p.identity.evaluation_id, p.identity)
            decision = resume_decision(p, inspection, consumed or {})
            if decision.startswith("BLOCK_") or decision in ("LEDGER_SCHEMA_READ_ONLY", "legacy_without_evaluation_ledger"):
                raise SchedulingBlocked(decision, p)
            from preparation_recovery import execution_spec
            execution_spec(self.ledger, p, inspection)
            if decision.startswith("BOUND_"):
                read_binding(self.ledger, p)
            rows.append((p, decision))
        return tuple(rows)

    def plan(self, entry, *, stage, candidate_id, policy_fingerprint, source_fingerprint,
             task_draw_index=None, selection_digest=None, ordinal=None):
        manifest = self.manifest
        return EvaluationPlan.create(run_id=manifest["run_id"], stage=stage, candidate_id=candidate_id,
            entry=entry, ordinal=len(self.plans.read()) if ordinal is None else ordinal,
            task_draw_index=task_draw_index, run_fingerprint=manifest["reproducibility_fingerprint"],
            task_schedule_identity=el.digest(manifest["tasks"]), policy_fingerprint=policy_fingerprint,
            source_fingerprint=source_fingerprint,
            model_fingerprint=manifest["evaluation_execution"]["model_fingerprint"],
            evaluator_fingerprint=manifest["evaluation_execution"]["evaluator_fingerprint"],
            selection_digest=selection_digest,
            search_identity=el.digest(manifest["experiment"]) if stage == "search" else None)

    def publish(self, plan):
        self.verify()
        self.plans.publish(plan)
        self.ledger.plan(plan.identity)

    def execute(self, plan, execute_one, *, independent=False):
        """Caller holds run lock; execute_one must use the existing lifecycle boundary."""
        check(plan.identity.stage != "checkpoint_bai" or (
            self.manifest.get("v2_derived", {}).get("schema_version") == 1 and
            self.manifest["v2_derived"].get("domain") == "bai" and
            self.manifest.get("experiment", {}).get("algorithm") == "wgm_v2"),
            "CHECKPOINT_BAI_NOT_IMPLEMENTED")
        if independent:
            from evaluation_batch import independent_plan
            # Only the fixed-batch wrapper uses this path, after freezing the
            # whole schedule under the run lease. No worker publishes plans or
            # inspects another worker's temporarily in-flight model marker.
            check(independent_plan(self, plan)
                  and plan in self.plans.read(), "INDEPENDENT_FROZEN_PLAN_REQUIRED")
        else:
            self.publish(plan)
            for earlier, decision in self.precheck():
                if earlier.identity.ordinal >= plan.identity.ordinal:
                    break
                if decision != "BOUND_PENDING_CONSUMPTION":
                    raise SchedulingBlocked("EARLIER_EVALUATION_INCOMPLETE", earlier)
        # Reconciliation sometimes advances only one local step. A finite bound,
        # with durable progress required, cannot become an automatic retry loop.
        for _ in range(5):
            inspection = self.ledger.inspect(plan.identity.evaluation_id, plan.identity)
            decision = resume_decision(plan, inspection, {})
            if decision in ("BOUND_PENDING_CONSUMPTION", "BOUND_FAILURE"):
                return read_binding(self.ledger, plan)
            if decision.startswith("BLOCK_"):
                raise SchedulingBlocked(decision, plan)
            from preparation_recovery import execution_spec
            spec = execution_spec(self.ledger, plan, inspection)
            before = tuple(e.sha256 for e in inspection.snapshot.events) if inspection.snapshot else ()
            outcome = (execute_one(plan, self.ledger) if spec == plan.spec() else
                       execute_one(plan, self.ledger, execution_spec=spec))
            if outcome.decision.startswith("BLOCK_"):
                raise SchedulingBlocked(outcome.decision, plan)
            after = self.ledger.inspect(plan.identity.evaluation_id, plan.identity)
            if after.error_code:
                raise SchedulingBlocked(after.error_code, plan)
            events = tuple(e.sha256 for e in after.snapshot.events) if after.snapshot else ()
            if events == before:
                raise SchedulingBlocked("SCHEDULER_NO_DURABLE_PROGRESS", plan)
        raise SchedulingBlocked("SCHEDULER_NO_TERMINAL_PROGRESS", plan)


def consume_binding(payload, result, config):
    """Pure copy-on-write transition. No files, global RNG, callbacks or external calls."""
    plan, i = result.plan, result.plan.identity
    check(i.stage == "search", "CONSUMER_STAGE")
    context = payload.get("execution")
    check(config.execution.policy == "validated", "CONSUMER_POLICY")
    check(context is not None and context.get("schema_version") == 1 and context["run_id"] == i.run_id
          and context["run_fingerprint"] == plan.run_fingerprint
          and context["scheduler_identity"] == SCHEDULER, "CONSUMER_IDENTITY")
    previous = context["consumed_bindings"].get(i.evaluation_id)
    if previous is not None:
        check(previous == result.binding_digest, "CONSUMER_BINDING_CONFLICT")
        return payload  # Exact no-op, including no checkpoint publication.
    check(result.outcome == "success" and type(result.resolved) is bool, "CONSUMER_REQUIRES_SUCCESS")
    check(isinstance(payload.get("inflight"), dict)
          and payload["inflight"].get("evaluation_plan") == asdict(plan)
          and payload["inflight"].get("task_id") == i.task_id, "CONSUMER_INFLIGHT")
    check(i.task_draw_index == payload["n_task_evals"], "CONSUMER_DRAW_INDEX")
    check(i.task_id in payload["tasks"], "CONSUMER_TASK")
    rows = [n for n in payload["nodes"] if n["commit_id"] == i.candidate_id]
    check(len(rows) == 1 and rows[0]["id"] == payload["inflight"]["node_id"], "CONSUMER_CANDIDATE")
    check(i.task_id in payload["submitted"][str(rows[0]["id"])], "CONSUMER_SUBMITTED")
    new = deepcopy(payload)
    node = next(n for n in new["nodes"] if n["id"] == rows[0]["id"])
    node["utility_measures"].append(int(result.resolved))
    if "legacy_list_topology" in new:
        from hgm_incremental import consume_aliases
        consume_aliases(new, node)
    # Keep the existing unique-task lookup for V1 scheduling; each binding still
    # has its own observation and valid-budget charge, including explicit repeats.
    new["results"].setdefault(i.candidate_id, {}).setdefault(i.task_id, int(result.resolved))
    new["n_task_evals"] += 1
    new["execution"]["consumed_bindings"][i.evaluation_id] = result.binding_digest
    new["execution"]["observations"][i.evaluation_id] = int(result.resolved)
    new["inflight"] = None
    if new["selector"] is not None:
        import numpy as np
        from wgm_selector import WGMSelector
        local_rng = np.random.RandomState(0)
        selector = WGMSelector.restore(new["selector"], config.wgm, local_rng)
        selector.observe(i.candidate_id, i.task_id, int(result.resolved))
        new["selector"] = selector.state_dict()
        new["summary"].update(selector.summary())
    new["summary"]["pending_task_deficit"] = sum(
        max(0, r["target_M"] - len(new["results"].get(r["candidate_id"], {})))
        for r in new["history"] if r["status"] == "admitted")
    new["summary"]["search_task_evaluations"] = new["n_task_evals"]
    return new


def commit_binding(directory, state, count, result, fingerprint, node_type, *, fault=lambda point: None):
    """The only consumption write is the canonical SearchState checkpoint."""
    from run_metadata import digest
    run = ExecutionRun(directory, el._read_json(Path(directory) / "run_manifest.json"))
    check(fingerprint == run.manifest["reproducibility_fingerprint"], "CONSUMER_FINGERPRINT")
    plans = run.plans.read()
    ordinal = result.plan.identity.ordinal
    check(ordinal < len(plans) and plans[ordinal] == result.plan, "CONSUMER_PLAN")
    check(read_binding(run.ledger, result.plan) == result, "CONSUMER_BINDING_CHANGED")
    payload = state.state_dict(count)
    updated = consume_binding(payload, result, state.config)
    if updated is payload:
        return count
    if state.execution.get("search_operation_schema") == 1:
        updated = el._decode(el.canonical(updated))
    envelope = {"state": updated, "sha256": digest(updated), "fingerprint": fingerprint}
    durable_json(Path(directory) / "search_state.json", envelope, replace_existing=True, fault=fault)
    # If persistence raised (even after rename), do not mark an in-memory success.
    # A fresh controller reads the single published checkpoint on resume.
    return state.restore(updated, node_type)
