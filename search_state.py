"""Corrected search bookkeeping shared by HGM and WGM; no external execution."""
from collections import Counter, defaultdict
from copy import deepcopy
import json
import math
from pathlib import Path
import random

import numpy as np

from search_config import experiment
from wgm_selector import WGMSelector, lineage


def lock_run(directory):
    """One writer owns a corrected run, including resumed runs."""
    import fcntl
    lease = (Path(directory) / "search.lock").open("a")
    try:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lease.close()
        raise ValueError("Another process owns this search run") from None
    return lease


def save_checkpoint(directory, state, n_task_evals, fingerprint, *, fault=lambda point: None):
    from run_metadata import atomic_json, digest
    payload = state.state_dict(n_task_evals)
    if state.execution is not None and state.execution.get("search_operation_schema") == 1:
        # JSON object keys are strings on restore. Normalize before hashing so
        # node 10+ cannot change the numeric-vs-lexical ordering of branch keys.
        payload = json.loads(json.dumps(payload, allow_nan=False))
    envelope = {"state": payload, "sha256": digest(payload), "fingerprint": fingerprint}
    if state.config.search.algorithm == "wgm_v2":
        envelope["runtime_control"] = deepcopy(state.runtime_control)
    if state.execution is not None:
        from evaluation_plan import durable_json
        durable_json(Path(directory) / "search_state.json", envelope, replace_existing=True, fault=fault)
    else:
        atomic_json(Path(directory) / "search_state.json", envelope)


def load_checkpoint(directory):
    from run_metadata import digest
    envelope = json.loads((Path(directory) / "search_state.json").read_text())
    if envelope["sha256"] != digest(envelope["state"]):
        raise ValueError("Search checkpoint checksum differs")
    return envelope


def validate_resume_profile(directory, config):
    from run_metadata import digest, fingerprint_inputs
    manifest = json.loads((Path(directory) / "run_manifest.json").read_text())
    saved = manifest.get("experiment")
    if saved is None:
        raise ValueError("Unlabelled historical search run cannot be resumed as a new profile")
    if (digest(manifest["fingerprint_inputs"]) != manifest["reproducibility_fingerprint"]
            or fingerprint_inputs(manifest) != manifest["fingerprint_inputs"]):
        raise ValueError("Saved manifest identity is inconsistent")
    count = saved.get("wgm", {}).get("effective_branch_count")
    if experiment(config, config.search.algorithm, count) != saved:
        raise ValueError("Resume algorithm/profile/search parameters differ")
    return count


def clade_observations(node, num_pseudo):
    """Pure version of the reference clade statistic; never call aliasing helper."""
    own = (list(node.utility_measures) if node.num_evals < num_pseudo
           else [node.mean_utility] * num_pseudo)
    for descendant in node.get_sub_tree()[1:]:
        own.extend(descendant.utility_measures)
    return own


def select_parent(nodes, sample, num_pseudo):
    """The single corrected clade TS implementation, with caller-restricted nodes."""
    return nodes[sample([clade_observations(node, num_pseudo) for node in nodes])]


def rng_state():
    name, keys, pos, gaussian, cached = np.random.get_state()
    return {"python": random.getstate(), "numpy": [name, keys.tolist(), pos, gaussian, cached]}


def restore_rng(state):
    def tuples(value):
        return tuple(tuples(v) for v in value) if isinstance(value, (list, tuple)) else value
    random.setstate(tuples(state["python"]))
    name, keys, pos, gaussian, cached = state["numpy"]
    np.random.set_state((name, np.array(keys, dtype=np.uint32), pos, gaussian, cached))


def beta_cdf(x, a, b):
    """Regularized incomplete beta via a continued fraction (positive a,b)."""
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    if x > (a+1)/(a+b+2):
        return 1-beta_cdf(1-x, b, a)
    tiny = 1e-300
    c, d = 1.0, 1-(a+b)*x/(a+1)
    d = 1/(d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 501):
        for aa in (m*(b-m)*x/((a+2*m-1)*(a+2*m)),
                   -(a+m)*(a+b+m)*x/((a+2*m)*(a+2*m+1))):
            d = 1+aa*d
            if abs(d) < tiny: d = tiny
            c = 1+aa/c
            if abs(c) < tiny: c = tiny
            d = 1/d
            delta = d*c
            h *= delta
        if abs(delta-1) < 2e-14:
            scale = math.exp(math.lgamma(a+b)-math.lgamma(a)-math.lgamma(b)
                             + a*math.log(x)+b*math.log1p(-x))
            return min(1.0, max(0.0, scale*h/a))
    raise ArithmeticError("Beta CDF did not converge")


def beta_quantile(success, failure, q=0.01):
    if type(success) is not int or type(failure) is not int or min(success, failure) < 0 or not 0 < q < 1:
        raise ValueError("Invalid Beta counts or quantile")
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = (lo+hi)/2
        if beta_cdf(mid, 1+success, 1+failure) < q: lo = mid
        else: hi = mid
    return (lo+hi)/2


def best_belief(nodes, *, post_hoc=False):
    rows = []
    for node in nodes:
        values = node.utility_measures
        if any(v not in (0, 1) for v in values):
            raise ValueError("Best-belief requires binary own observations")
        successes = int(sum(values))
        failures = len(values)-successes
        rows.append({"node_id": node.id, "candidate_id": node.commit_id,
                     "successes": successes, "failures": failures,
                     "score": beta_quantile(successes, failures)})
    winner = min(rows, key=lambda row: (-row["score"], row["node_id"])) if rows else None
    return {"method": "hgm_best_belief", "epsilon_percent": 1.0, "q": 0.01,
            "convention": "provisional_lower_quantile", "read_only": True,
            "metric_kind": "post_hoc_metric" if post_hoc else "final_report",
            "count_provenance": "legacy_node_state_may_include_aliasing" if post_hoc else "own_unique_task_results",
            "selected": winner, "scores": rows}


class SearchState:
    def __init__(self, config, nodes, submitted, tasks):
        self.config, self.nodes, self.submitted = config, nodes, submitted
        self.tasks = list(tasks)
        self.selector = WGMSelector(config.wgm, np.random) if config.search.algorithm == "wgm" else None
        self.branches = self.selector.branches if self.selector else {0: 0}
        self.initial_attempts = 0
        self.initialized = False
        self.history = []
        self.results = defaultdict(dict, {n.commit_id: dict(getattr(n, "completed_task_results", {}))
                                          for n in nodes.values()})
        self.inflight = None
        self.execution = None

    def identity(self):
        count = len(set(self.branches.values())) if self.initialized else None
        return experiment(self.config, self.config.search.algorithm, count)

    def deficit(self):
        if self.execution is not None and self.execution.get("search_operation_schema") == 1:
            # N2 snapshots must be read-only, including missing child results.
            return sum(max(0, r["target_M"]-len(self.results.get(r["candidate_id"], {})))
                       for r in self.history if r["status"] == "admitted")
        return sum(max(0, r["target_M"]-len(self.results[r["candidate_id"]]))
                   for r in self.history if r["status"] == "admitted")

    def can_expand(self, remaining):
        target = (self.config.wgm.reward_batch_size if self.selector and self.initialized else 1)
        return len(self.tasks) >= target and remaining >= self.deficit()+target

    def begin_expansion(self, nodes, sample, num_pseudo):
        if self.inflight:
            raise ValueError("Unresolved external operation")
        branch, probability = None, None
        if not self.initialized:
            nodes = [n for n in nodes if n.id == 0]
        elif self.selector:
            branch, probability = self.selector.select(self.branches[n.id] for n in nodes)
            nodes = [n for n in nodes if self.branches[n.id] == branch]
        if not nodes:
            raise ValueError("No expandable parent")
        parent = select_parent(nodes, sample, num_pseudo)
        branch = self.branches[parent.id]
        expansion_id = f"expansion-{len(self.history):06d}"
        candidate_id = f"candidate-{len(self.history):06d}"
        record = {"expansion_id": expansion_id, "candidate_id": candidate_id,
                  "parent_id": parent.id, "selected_branch_id": branch,
                  "selection_probability": probability, "child_id": None,
                  "phase": "search" if self.initialized else "initialization",
                  "target_M": self.config.wgm.reward_batch_size if self.selector and self.initialized else 1,
                  "status": "creating"}
        if self.initialized and self.selector:
            self.selector.begin(expansion_id, branch, parent.id, candidate_id, probability)
        self.history.append(record)
        self.inflight = {"kind": "expansion", "expansion_id": expansion_id}
        return parent, candidate_id

    def end_expansion(self, child):
        record = self.history[-1]
        initial = record["phase"] == "initialization"
        if child is None:
            record["status"] = "failed"
            if not initial and self.selector:
                self.selector.invalid(record["expansion_id"])
        else:
            record.update(status="admitted", child_id=child.id)
            if child.commit_id != record["candidate_id"]:
                raise ValueError("Candidate identity differs from expansion")
            if self.selector:
                self.selector.admit(child.id, child.parent_id, initial=initial)
                if not initial:
                    self.selector.feedback[record["expansion_id"]]["child_id"] = child.id
            else:
                self.branches[child.id] = len(set(self.branches.values())) if initial else self.branches[child.parent_id]
        self.initial_attempts += int(initial)
        self.inflight = None

    def freeze(self):
        if self.selector: self.selector.freeze()
        self.initialized = True

    def begin_evaluation(self, node, task):
        if self.inflight: raise ValueError("Unresolved external operation")
        self.inflight = {"kind": "evaluation", "node_id": node.id, "task_id": task}

    def observe(self, node, task, result):
        if task not in self.tasks or type(result) not in (int, bool) or result not in (0, 1):
            raise ValueError("Invalid unique search-task observation")
        previous = self.results[node.commit_id]
        if task not in previous:
            previous[task] = int(result)
            node.utility_measures.append(int(result))
            if self.selector: self.selector.observe(node.commit_id, task, int(result))
        elif previous[task] != int(result):
            raise ValueError("Conflicting repeated task result")
        self.inflight = None

    def summary(self):
        metrics = self.selector.summary() if self.selector else {
            "reproductive_lineage_L": lineage(r["selected_branch_id"] for r in self.history if r["phase"] == "search"),
            "design_diversity_D": None, "design_diversity_reason": "feature_schema_not_configured"}
        return {"experiment": self.identity(), "initial_attempts": self.initial_attempts,
                "initialization_complete": self.initialized, "pending_task_deficit": self.deficit(),
                "expansion_count": len(self.history), **metrics}

    def state_dict(self, n_task_evals):
        from hgm_incremental_config import enabled
        topology = {}
        if enabled(self.config):
            from hgm_incremental import list_topology
            topology["legacy_list_topology"] = list_topology(self.nodes)
        return {"schema_version": 1, "experiment": self.identity(), "rng": rng_state(),
                **topology,
                **({"execution": deepcopy(self.execution)} if self.execution is not None else {}),
                "nodes": [{"id": n.id, "commit_id": n.commit_id, "parent_id": n.parent_id,
                           "utility_measures": list(n.utility_measures)} for n in self.nodes.values()],
                "submitted": {str(k): sorted(v) for k,v in self.submitted.items()},
                "tasks": self.tasks, "n_task_evals": n_task_evals,
                "initial_attempts": self.initial_attempts, "initialized": self.initialized,
                "branches": dict(self.branches), "results": deepcopy(dict(self.results)),
                "history": deepcopy(self.history), "inflight": deepcopy(self.inflight),
                "selector": self.selector.state_dict() if self.selector else None,
                "summary": {**self.summary(), "search_task_evaluations": n_task_evals,
                            "initial_root_task_results": self.nodes[0].num_evals,
                            "root_cache_outside_search_budget": True}}

    def restore(self, state, node_type):
        from hgm_incremental_config import enabled
        if enabled(self.config):
            from hgm_incremental import validate_topology
            validate_topology(state)
        if (state.get("execution") or {}).get("search_operation_schema") == 1:
            # Validate before touching registry, selector or either global RNG.
            from search_operation_journal import validate_rng
            from search_operations import checkpoint_rng
            validate_rng(checkpoint_rng(state))
        if state.get("execution") is not None and self.config.execution.policy != "validated":
            raise ValueError("Search checkpoint execution policy differs")
        if state["tasks"] != self.tasks:
            raise ValueError("Resume search task order differs")
        count = len(set(state["branches"].values())) if state["initialized"] else None
        if state["experiment"] != experiment(self.config, self.config.search.algorithm, count):
            raise ValueError("Resume algorithm/profile/effective parameters differ")
        self.nodes.clear()
        for row in state["nodes"]:
            node_type(**row)
        for node in self.nodes.values():
            if node.parent_id is not None: self.nodes[node.parent_id].add_child(node)
        if enabled(self.config):
            from hgm_incremental import restore_aliases
            restore_aliases(state, self.nodes)
        self.submitted.clear()
        self.submitted.update({int(k):set(v) for k,v in state["submitted"].items()})
        self.initial_attempts, self.initialized = state["initial_attempts"], state["initialized"]
        self.results = defaultdict(dict, deepcopy(state["results"]))
        self.history, self.inflight = deepcopy(state["history"]), deepcopy(state["inflight"])
        self.execution = deepcopy(state.get("execution"))
        if self.selector:
            private = (self.execution is not None and self.execution.get("search_operation_schema") == 1
                       and self.execution.get("selector_rng_mode") == "private_numpy")
            self.selector = WGMSelector.restore(state["selector"], self.config.wgm,
                                                np.random.RandomState(0) if private else np.random)
            self.branches = self.selector.branches
        else:
            self.branches = {int(k):v for k,v in state["branches"].items()}
        restore_rng(state["rng"])
        return state["n_task_evals"]

    def reconcile(self, directory, node_type, n_task_evals):
        """Only consume durable terminal metadata; unknown operations fail closed."""
        if self.inflight is None:
            return n_task_evals
        if self.inflight["kind"] == "expansion":
            row = self.history[-1]
            folder = Path(directory) / row["candidate_id"]
            path = folder / "metadata.json"
            data = json.loads(path.read_text()) if path.is_file() else {}
            if data.get("status") == "success" and (folder / "model_patch.diff").is_file():
                child = node_type(row["candidate_id"], parent_id=row["parent_id"])
                self.nodes[child.parent_id].add_child(child)
                self.end_expansion(child)
                return n_task_evals
            if data.get("status") == "failed" and data.get("reason"):
                self.end_expansion(None)
                return n_task_evals
        else:
            node = self.nodes[self.inflight["node_id"]]
            task = self.inflight["task_id"]
            path = Path(directory) / node.commit_id / "metadata.json"
            data = json.loads(path.read_text()) if path.is_file() else {}
            performance = data.get("overall_performance", {})
            if task in performance.get("total_submitted_ids", []):
                already = task in self.results[node.commit_id]
                self.observe(node, task, int(task in performance.get("total_resolved_ids", [])))
                return n_task_evals + int(not already)
        raise ValueError("Ambiguous interrupted search operation; no new model calls are allowed")
