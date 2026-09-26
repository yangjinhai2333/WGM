"""Serial validated execution around the existing HGM/WGM selection primitives."""
from copy import deepcopy
from dataclasses import asdict
import random

import numpy as np

from evaluation_plan import EvaluationPlan, SCHEDULER, check
from evaluation_scheduler import SchedulingBlocked, commit_binding
from search_config import is_corrected, optimization
from search_state import SearchState, save_checkpoint, load_checkpoint


class ValidatedSearch:
    """Owns no second consumption ledger; all algorithm/bookkeeping is SearchState.

    callbacks: make_plan(node, task, draw_index), execute(plan, ledger),
    generate(parent, candidate_id), refresh_views(). ExecutionRun's lease is held
    for the entire search. Adapters remain the sole external-call authority.
    """
    def __init__(self, run, state, node_type, *, make_plan, execute, generate,
                 count=0, refresh_views=lambda: None, fault=lambda point: None, operations=None):
        self.run, self.state, self.node_type = run, state, node_type
        self.make_plan, self.execute_one, self.generate = make_plan, execute, generate
        self.refresh_views, self.fault, self.count = refresh_views, fault, count
        self.config = state.config
        self.opt = optimization(self.config)
        self.fingerprint = run.manifest["reproducibility_fingerprint"]
        expected = {"schema_version": 1, "run_id": run.manifest["run_id"], "run_fingerprint": self.fingerprint,
                    "scheduler_identity": SCHEDULER}
        if state.execution is None:
            state.execution = {**expected, "consumed_bindings": {}, "observations": {}}
        check(all(state.execution.get(k) == v for k, v in expected.items()), "SEARCH_EXECUTION_IDENTITY")
        self.operations = operations
        if operations is not None:
            operations.attach(self)

    def _validate_consumption(self):
        consumed, observations = self.state.execution["consumed_bindings"], self.state.execution["observations"]
        check(set(consumed) == set(observations) and len(consumed) == self.count, "SEARCH_CONSUMPTION_COUNT")
        check(all(type(v) is int and v in (0, 1) for v in observations.values()), "SEARCH_OBSERVATION_INVALID")

    def save(self):
        save_checkpoint(self.run.directory, self.state, self.count, self.fingerprint)

    def restore(self):
        if self.operations is not None:
            from search_operations import read_checkpoint
            checkpoint = read_checkpoint(self.run.directory)
        else:
            checkpoint = load_checkpoint(self.run.directory)
        check(checkpoint["fingerprint"] == self.fingerprint, "SEARCH_FINGERPRINT")
        check(checkpoint["state"].get("execution") is not None, "SEARCH_LEGACY_RESUME")
        self.count = self.state.restore(checkpoint["state"], self.node_type)
        if self.operations is not None:
            self.operations.attach(self)
        check(self.state.execution["run_id"] == self.run.manifest["run_id"]
              and self.state.execution.get("schema_version") == 1
              and self.state.execution["run_fingerprint"] == self.fingerprint
              and self.state.execution["scheduler_identity"] == SCHEDULER, "SEARCH_EXECUTION_IDENTITY")
        self._validate_consumption()

    def _sample(self, observations):
        a = [1 + np.sum(v) for v in observations]
        b = [1 + len(v) - np.sum(v) for v in observations]
        if self.opt.cool_down:
            budget = self.config.execution.max_task_evals
            factor = 10000 if self.count == budget else budget**self.opt.beta / (budget-self.count)**self.opt.beta
            a, b = np.array(a) * factor, np.array(b) * factor
        return np.argmax(np.random.beta(a, b))

    def _finish_evaluation(self, plan):
        result = self.run.execute(plan, self.execute_one)
        self.fault("binding.before_consumption")
        if result.outcome != "success":
            raise SchedulingBlocked("BOUND_FAILURE", plan)
        self.count = commit_binding(self.run.directory, self.state, self.count, result,
                                    self.fingerprint, self.node_type, fault=self.fault)
        self.fault("consumption.after_checkpoint")
        self.refresh_views()

    def reconcile(self):
        self._validate_consumption()
        pending = self.state.inflight
        # The checkpoint is published before the plan journal, so a crash in that
        # gap can only publish the exact persisted plan; no node/task sampling.
        if pending and pending["kind"] == "evaluation":
            check("evaluation_plan" in pending, "SEARCH_LEGACY_INFLIGHT")
            self.run.publish(EvaluationPlan.from_dict(pending["evaluation_plan"]))
        rows = self.run.precheck(self.state.execution["consumed_bindings"])
        planned = {p.identity.evaluation_id for p, _ in rows if p.identity.stage == "search"}
        check(set(self.state.execution["consumed_bindings"]).issubset(planned), "SEARCH_UNKNOWN_CONSUMPTION")
        if pending and pending["kind"] == "expansion":
            # Candidate metadata proves source provenance, not the post-diagnosis
            # RNG state. Generation has no evaluation ledger; never guess that
            # state or repeat a possibly completed model call.
            if self.operations is None:
                raise SchedulingBlocked("SEARCH_EXPANSION_RNG_UNCONFIRMED")
        if self.operations is not None:
            self.operations.reconcile(self)
        if pending and pending["kind"] == "evaluation":
            self._finish_evaluation(EvaluationPlan.from_dict(pending["evaluation_plan"]))
        for plan, decision in rows:
            if plan.identity.stage != "search":
                continue
            consumed = self.state.execution["consumed_bindings"]
            if plan.identity.evaluation_id not in consumed:
                raise SchedulingBlocked("SEARCH_UNCONSUMED_WITHOUT_INFLIGHT", plan)
        self.refresh_views()

    def _expand(self):
        state = self.state
        checkpoint = None
        if self.operations is not None:
            from search_operations import read_checkpoint
            check(state.inflight is None, "SEARCH_PENDING_OPERATION")
            checkpoint = read_checkpoint(self.run.directory)
        try:
            parent, cid = self._select_expansion()
        except BaseException:
            if self.operations is not None:
                self.restore()
            raise
        if self.operations is not None:
            try:
                plan = self.operations.plan(self, parent, cid, checkpoint)
                state.inflight["operation_plan"] = plan.to_dict()
                self.operations.fault("operation.before_checkpoint")
                save_checkpoint(self.run.directory, self.state, self.count, self.fingerprint,
                                fault=lambda point: self.operations.fault("operation.plan." + point))
            except BaseException:
                self.restore()
                raise
            self.operations.fault("operation.after_checkpoint")
            self.operations.journal.publish(plan)
            self.operations.fault("operation.after_plan")
            terminal = self.operations.execute(plan, self)
            self.operations.commit(self, plan, terminal)
            return
        self.save()
        child = self.generate(parent, cid)  # A crash here cannot repeat generation on resume.
        state.end_expansion(child)
        self.save()

    def _select_expansion(self):
        state = self.state
        nodes = [n for n in state.nodes.values() if np.isfinite(n.mean_utility) and n.mean_utility > 0]
        if not nodes:
            raise SchedulingBlocked("SEARCH_NO_EXPANDABLE_PARENT")
        if is_corrected(self.config):
            parent, cid = state.begin_expansion(nodes, self._sample, self.opt.n_pseudo_descendant_evals)
        else:
            # Retain legacy clade aliasing; validated changes failure policy,
            # not the explicitly selected historical HGM profile.
            parent = nodes[self._sample([n.get_decendant_evals(self.opt.n_pseudo_descendant_evals) for n in nodes])]
            index = len(state.history)
            cid, eid = f"candidate-{index:06d}", f"expansion-{index:06d}"
            state.history.append({"expansion_id": eid, "candidate_id": cid, "parent_id": parent.id,
                "selected_branch_id": state.branches[parent.id], "selection_probability": None,
                "child_id": None, "phase": "search" if state.initialized else "initialization",
                "target_M": 1, "status": "creating"})
            state.inflight = {"kind": "expansion", "expansion_id": eid}
        return parent, cid

    def run_to_budget(self, *, target_budget=None):
        if target_budget is not None:
            from hgm_incremental_config import enabled
            check(enabled(self.config) and type(target_budget) is int
                  and self.count <= target_budget <= self.config.hgm_incremental.target_budget
                  <= self.config.execution.max_task_evals, "HGM_PAUSE_TARGET")
        self.reconcile()
        budget = self.config.execution.max_task_evals if target_budget is None else target_budget
        target = (5 if is_corrected(self.config) else
                  min(5, int(self.config.execution.max_workers**self.opt.alpha)))
        while not self.state.initialized:
            if self.state.initial_attempts < target and (not is_corrected(self.config)
                    or self.state.can_expand(budget - self.count)):
                self._expand()
            else:
                self.state.freeze()
                self.save()
        while self.count < budget:
            random.random()  # Preserve the legacy/corrected jitter draw, without sleeping.
            state = self.state
            expandable = any(np.isfinite(n.mean_utility) and n.mean_utility > 0 for n in state.nodes.values())
            should_expand = self.count**self.opt.alpha >= len(state.nodes) - 1
            if is_corrected(self.config):
                should_expand = should_expand and expandable and state.can_expand(budget - self.count)
            if should_expand:
                self._expand()
                continue
            nodes = [n for n in state.nodes[0].get_sub_tree() if len(state.submitted[n.id]) < len(state.tasks)]
            if not nodes:
                raise SchedulingBlocked("SEARCH_TASK_POOL_EXHAUSTED")
            node = nodes[self._sample([n.utility_measures for n in nodes])]
            available = [t for t in state.tasks if t not in state.submitted[node.id]]
            task = random.choice(available) if random.random() < self.opt.eval_random_level else available[0]
            plan = self.make_plan(node, task, self.count)
            state.submitted[node.id].add(task)
            state.begin_evaluation(node, task)
            state.inflight["evaluation_plan"] = asdict(plan)
            self.save()  # Includes advanced RNG and the exact identity before any preparation.
            self.fault("planning.after_checkpoint")
            self.run.publish(plan)
            self._finish_evaluation(plan)
        return self.count
