"""Budget pauses around validated legacy HGM; no selector or external-call implementation."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import evaluation_ledger as io
from evaluation_plan import check, durable_json
from evaluation_scheduler import read_binding
from hgm_incremental_config import enabled, validate
from run_metadata import digest
from search_operations import read_checkpoint


def list_topology(nodes):
    groups, by_object = [], {}
    for node in nodes.values():
        key = id(node.utility_measures)
        if key not in by_object:
            by_object[key] = len(groups)
            groups.append([])
        groups[by_object[key]].append(node.id)
    return {"utility_aliases": groups,
            "children": {str(n.id): [c.id for c in n.children] for n in nodes.values()}}


def validate_topology(state):
    topology = state.get("legacy_list_topology")
    check(isinstance(topology, dict), "HGM_ALIAS_TOPOLOGY_MISSING")
    nodes = {n["id"]: n for n in state["nodes"]}
    groups = topology["utility_aliases"]
    flat = [n for group in groups for n in group]
    check(len(flat) == len(nodes) and set(flat) == set(nodes) and all(groups), "HGM_ALIAS_TOPOLOGY")
    for group in groups:
        check(all(nodes[n]["utility_measures"] == nodes[group[0]]["utility_measures"] for n in group),
              "HGM_ALIAS_VALUES")
    children = topology["children"]
    check(set(children) == {str(n) for n in nodes}, "HGM_CHILDREN")
    for nid in nodes:
        # Legacy attaches each new child once, in registry/admission order.
        check(children[str(nid)] == [n["id"] for n in state["nodes"] if n["parent_id"] == nid],
              "HGM_CHILDREN")
    check(list(nodes) == list(range(len(nodes))) and nodes[0]["parent_id"] is None
          and all(n["parent_id"] in nodes and n["parent_id"] < n["id"]
                  for n in state["nodes"][1:]), "HGM_TREE")


def restore_aliases(state, nodes):
    for group in state["legacy_list_topology"]["utility_aliases"]:
        shared = nodes[group[0]].utility_measures
        for nid in group[1:]:
            nodes[nid].utility_measures = shared


def consume_aliases(state, changed_node):
    """JSON loses references: propagate exactly the append on the original shared list."""
    nodes = {n["id"]: n for n in state["nodes"]}
    for group in state["legacy_list_topology"]["utility_aliases"]:
        if changed_node["id"] in group:
            for nid in group:
                nodes[nid]["utility_measures"] = list(changed_node["utility_measures"])


def check_evidence(run, row, root_results):
    state, context = row["state"], row["state"]["execution"]
    validate_topology(state)
    consumed, observations = context["consumed_bindings"], context["observations"]
    check(set(consumed) == set(observations) and len(consumed) == state["n_task_evals"], "HGM_DIRECT_COUNT")
    plans = {p.identity.evaluation_id: p for p in run.plans.read()}
    direct = {"initial": root_results}
    for eid, binding in consumed.items():
        check(eid in plans, "HGM_DIRECT_PLAN")
        plan = plans[eid]
        result = read_binding(run.ledger, plan)
        check(plan.identity.stage == "search" and result.outcome == "success"
              and result.binding_digest == binding and observations[eid] == int(result.resolved), "HGM_DIRECT_BINDING")
        own = direct.setdefault(plan.identity.candidate_id, {})
        check(plan.identity.task_id not in own, "HGM_DIRECT_DUPLICATE")
        own[plan.identity.task_id] = int(result.resolved)
    ids = {n["commit_id"] for n in state["nodes"]}
    check(set(direct) <= ids and {c: r for c, r in state["results"].items() if r} == direct,
          "HGM_DIRECT_RESULTS")
    return direct


def read_snapshot(path, fingerprint):
    row = io._read_json(path)
    check(row["snapshot_digest"] == digest({k: v for k, v in row.items() if k != "snapshot_digest"})
          and row["sha256"] == digest(row["state"]) and row["fingerprint"] == fingerprint,
          "HGM_SNAPSHOT_DIGEST")
    validate_topology(row["state"])
    check(row["algorithm"] == "hgm" and row["profile"] == "legacy_code"
          and row["planning_horizon"] == row["state"]["experiment"]["incremental"]["planning_horizon"]
          and row["valid_search_budget"] == row["state"]["n_task_evals"]
          <= row["requested_target"] <= row["planning_horizon"], "HGM_SNAPSHOT_BUDGET")
    selected = row["final_selection"]
    check(selected["selection_digest"] == digest({k: v for k, v in selected.items() if k != "selection_digest"})
          and selected["checkpoint_digest"] == row["sha256"]
          and selected["checkpoint"] == row["state"]["n_task_evals"], "HGM_SELECTION_DIGEST")
    return row


def read_control(directory, config, manifest):
    """Read-only preflight guard. Runtime target never enters N2 plan/checkpoint identities."""
    validate(config)
    cfg = config.hgm_incremental
    check(enabled(config) and config.execution.max_task_evals == cfg.planning_horizon, "HGM_HORIZON")
    check(manifest["experiment"].get("incremental") == cfg.identity(), "HGM_STANDALONE_NOT_INCREMENTAL")
    path = Path(directory) / "hgm_control.json"
    state_path = Path(directory) / "search_state.json"
    checkpoint = read_checkpoint(directory) if state_path.exists() else None
    if checkpoint:
        check(checkpoint["fingerprint"] == manifest["reproducibility_fingerprint"]
              and checkpoint["state"]["experiment"] == manifest["experiment"], "HGM_CHECKPOINT_IDENTITY")
        validate_topology(checkpoint["state"])
    if not path.exists():
        check(checkpoint is None or (checkpoint["state"]["n_task_evals"] == 0
              and not checkpoint["state"]["history"] and checkpoint["state"]["inflight"] is None),
              "HGM_CONTROL_MISSING")
        return None
    row = io._read_json(path)
    control = row["control"]
    check(row["sha256"] == digest(control) and control["schema_version"] == 1
          and control["run_fingerprint"] == manifest["reproducibility_fingerprint"]
          and control["planning_horizon"] == cfg.planning_horizon, "HGM_CONTROL_IDENTITY")
    check(cfg.target_budget >= control["requested_target"], "HGM_TARGET_DECREASE")
    check(checkpoint is not None and control["completed_valid_budget"] <= checkpoint["state"]["n_task_evals"]
          <= control["requested_target"] <= cfg.planning_horizon, "HGM_CONTROL_BUDGET")
    for budget, snapshot_digest in control["completed_checkpoints"].items():
        snapshot = read_snapshot(Path(directory) / "checkpoints" / f"{int(budget):06d}.json",
                                 manifest["reproducibility_fingerprint"])
        check(snapshot["state"]["n_task_evals"] == int(budget)
              and snapshot["snapshot_digest"] == snapshot_digest, "HGM_CHECKPOINT_CHANGED")
    if control["status"] == "completed":
        check(checkpoint["state"]["inflight"] is None
              and checkpoint["state"]["n_task_evals"] == control["requested_target"]
              and checkpoint["sha256"] == control["checkpoint_digest"], "HGM_COMPLETED_CHECKPOINT_CHANGED")
    return control


class IncrementalHGM:
    def __init__(self, engine, root_results, *, wall_seconds=lambda: 0.0, fault=lambda point: None):
        self.engine, self.run, self.config = engine, engine.run, engine.config
        self.root_results, self.wall_seconds, self.fault = root_results, wall_seconds, fault
        self.control = read_control(self.run.directory, self.config, self.run.manifest)

    def save_control(self):
        durable_json(self.run.directory / "hgm_control.json",
                     {"control": self.control, "sha256": digest(self.control)}, replace_existing=True)
        self.run.manifest["hgm_runtime"] = deepcopy(self.control)
        durable_json(self.run.directory / "run_manifest.json", self.run.manifest, replace_existing=True)

    def verify(self):
        row = read_checkpoint(self.run.directory)
        check_evidence(self.run, row, self.root_results)
        consumed_evals = self.engine.state.execution["consumed_bindings"]
        rows = self.run.precheck(consumed_evals)
        pending = self.engine.state.inflight or {}
        allowed = set(consumed_evals)
        if pending.get("kind") == "evaluation":
            allowed.add(pending["evaluation_plan"]["identity"]["evaluation_id"])
        check(all(p.identity.evaluation_id in allowed for p, _ in rows if p.identity.stage == "search"),
              "HGM_UNCLAIMED_SEARCH_PLAN")
        operations = self.engine.operations
        check(operations is not None, "HGM_N2_REQUIRED")
        consumed = self.engine.state.execution["consumed_operations"]
        plans = operations.journal.plans()
        check(set(consumed) <= {p.operation_id for p in plans}, "HGM_OPERATION_MISSING")
        for plan in plans:
            state, _ = operations.journal.inspect(plan)
            committed = plan.operation_id in consumed
            operations.verify(plan, self.engine, committed=committed)
            if committed:
                terminal = state["terminal"]
                check(terminal is not None and consumed[plan.operation_id] == {
                    "operation_terminal_digest": terminal["terminal_digest"],
                    "candidate_digest": terminal["candidate_digest"]}, "HGM_OPERATION_COMMIT")

    def snapshot(self):
        from final_adapter import choose_legacy, selection_evidence
        engine = self.engine
        row = read_checkpoint(self.run.directory)
        check(engine.state.inflight is None and engine.count == row["state"]["n_task_evals"], "HGM_PAUSE_INFLIGHT")
        direct = check_evidence(self.run, row, self.root_results)
        folder = self.run.directory / "checkpoints"
        folder.mkdir(exist_ok=True)
        path = folder / f"{engine.count:06d}.json"
        if path.exists():
            frozen = read_snapshot(path, engine.fingerprint)
            check(frozen["sha256"] == row["sha256"], "HGM_SNAPSHOT_CHANGED")
            return frozen
        selected = choose_legacy("hgm", [n["commit_id"] for n in row["state"]["nodes"]], direct)
        source = engine.operations.sources.get(selected["candidate_id"])
        row["final_selection"] = selection_evidence(SimpleNamespace(digest=row["sha256"], b=engine.count), selected, source)
        row.update(algorithm="hgm", profile="legacy_code", planning_horizon=self.config.hgm_incremental.planning_horizon,
                   requested_target=self.config.hgm_incremental.target_budget, valid_search_budget=engine.count,
                   root_identity=self.run.manifest["hgm_incremental_inputs"])
        row["search_wall_time_seconds"] = self.wall_seconds()
        row["snapshot_digest"] = digest(row)
        durable_json(path, row, fault=lambda point: self.fault("snapshot." + point))
        return row

    def run_to_target(self):
        self.verify()  # Every journal and binding is checked before any possible external work.
        cfg, engine = self.config.hgm_incremental, self.engine
        if self.control is not None and self.control["status"] == "completed" and cfg.target_budget == self.control["requested_target"]:
            return engine.count
        previous_target = self.control["requested_target"] if self.control else cfg.target_budget
        if self.control is None:
            self.control = {"schema_version": 1, "run_fingerprint": engine.fingerprint,
                "planning_horizon": cfg.planning_horizon, "requested_target": cfg.target_budget,
                "completed_valid_budget": 0, "completed_checkpoints": {}, "continuation_history": [],
                "status": "running", "canonical_search_wall_seconds": 0, "checkpoint_digest": None}
        if not self.control["continuation_history"] or cfg.target_budget != self.control["requested_target"]:
            self.control["continuation_history"].append({"from_valid_budget": engine.count,
                "target_budget": cfg.target_budget})
        self.control.update(requested_target=cfg.target_budget, status="running")
        self.control["configured_evaluation_workers"] = self.config.execution.evaluation_workers
        self.save_control()
        stops = sorted({b for b in cfg.checkpoints if engine.count < b <= cfg.target_budget} | {cfg.target_budget})
        if engine.count <= previous_target < cfg.target_budget and str(previous_target) not in self.control["completed_checkpoints"]:
            stops = sorted(set(stops) | {previous_target})
        for target in stops:
            engine.run_to_budget(target_budget=target)
            self.fault("pause.before_snapshot")
            frozen = self.snapshot()
            self.fault("pause.after_snapshot")
            self.control["completed_checkpoints"][str(target)] = frozen["snapshot_digest"]
            self.control.update(completed_valid_budget=engine.count, checkpoint_digest=frozen["sha256"],
                canonical_search_wall_seconds=frozen["search_wall_time_seconds"],
                status="completed" if target == cfg.target_budget else "running")
            self.save_control()
        return engine.count


def report(directory, manifest):
    from final_adapter import search_accounting
    from hgm_incremental_config import HGMIncrementalConfig
    from final_adapter import config_from_manifest
    saved = io._read_json(Path(directory) / "hgm_control.json")["control"]
    config = config_from_manifest(manifest)
    config.hgm_incremental = HGMIncrementalConfig(target_budget=saved["requested_target"])
    control = read_control(directory, config, manifest)
    row = read_checkpoint(directory)
    budget = row["state"]["n_task_evals"]
    cp = SimpleNamespace(directory=Path(directory), manifest=manifest, state=row["state"], row=row, b=budget)
    stats = search_accounting(cp)
    if control["status"] == "completed":
        stats["search_wall_seconds"] = control["canonical_search_wall_seconds"]
    summary_path = Path(directory) / "run_summary.json"
    status = control["status"]
    if status != "completed" and summary_path.exists():
        status = io._read_json(summary_path).get("status", status)
    return {"report_type": "hgm_incremental_search", "algorithm": "hgm", "profile": "legacy_code",
        **deepcopy(control), "completed_checkpoints": sorted(map(int, control["completed_checkpoints"])),
        "status": status, "completed_valid_budget": budget,
        "canonical_search_wall_seconds": stats["search_wall_seconds"],
        "valid_search_budget": budget, "checkpoint_digest": row["sha256"],
        "root_digest": manifest["root_initialization"]["root_cache_digest"],
        "root_identity": manifest.get("hgm_incremental_inputs"), "search": stats,
        "evaluation_workers": control["configured_evaluation_workers"], "effective_search_workers": 1,
        "final_selection_rule": "empirical_mean_then_sample_count_then_candidate_id"}


def report_text(row):
    return "\n".join(["HGM legacy_code incremental search", f"Status: {row['status']}",
        f"Horizon: {row['planning_horizon']}; requested target: {row['requested_target']}; valid budget: {row['valid_search_budget']}",
        f"Completed checkpoints: {row['completed_checkpoints']}",
        f"Checkpoint digest: {row['checkpoint_digest']}",
        f"Search calls/timing: {row['search']}", "Search workers: 1; final60 is explicit and outside search budget."])
