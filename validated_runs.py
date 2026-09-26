"""Validated baseline/search/final entry points. No work occurs at import time."""
from collections import defaultdict
from copy import deepcopy
from dataclasses import replace
from functools import partial
import hashlib
import json
from pathlib import Path
import random
import tempfile
import threading

import evaluation_ledger as el
from task_schedule import pool_size
from evaluation_plan import check, durable_json, model_fingerprint
from evaluation_scheduler import ExecutionRun, SchedulingBlocked, read_binding
from execution_policy import check_resume, require_supported


def _open(config, directory, proposed, resume):
    require_supported(config)
    directory = el._checked_path(directory)
    if resume:
        check_resume(directory, config)
        # Never bootstrap missing ledgers, including directories with an old manifest.
        check((directory / "evaluation_ledger/run.json").is_file(), "LEGACY_WITHOUT_EVALUATION_LEDGER")
        saved = el._read_json(directory / "run_manifest.json")
        check(saved["reproducibility_fingerprint"] == proposed["reproducibility_fingerprint"], "RUN_RESUME_FINGERPRINT")
        return ExecutionRun(directory, saved)
    check(not directory.exists(), "RUN_DIRECTORY_EXISTS")
    check(directory.parent.is_dir(), "RUN_PARENT_MISSING")
    return ExecutionRun.create(directory, proposed)


def _entry_map(config, tasks):
    from downstream_solver import load_swe_dataset, select_entries
    if config.evaluation.polyglot:
        import json
        from pathlib import Path as _Path
        dataset = json.loads(_Path(config.evaluation.polyglot_dataset_path).read_text())
    else:
        dataset = load_swe_dataset(config.evaluation.dataset_name, config.evaluation.dataset_path)
    return {e["instance_id"]: dict(e) for e in select_entries(dataset, tasks)}


def _check_search_checkpoint(run):
    if (run.directory / "search_state.json").exists():
        from search_state import load_checkpoint
        checkpoint = load_checkpoint(run.directory)
        context = checkpoint["state"].get("execution", {})
        check(checkpoint["fingerprint"] == run.manifest["reproducibility_fingerprint"]
              and context.get("schema_version") == 1
              and context.get("run_id") == run.manifest["run_id"]
              and context.get("run_fingerprint") == checkpoint["fingerprint"]
              and context.get("scheduler_identity") == "validated_scheduler_v1", "SEARCH_EXECUTION_IDENTITY")


def _check_configuration(config, run):
    from run_metadata import effective_configuration, fingerprint_inputs
    proposed = deepcopy(run.manifest)
    proposed["configuration"]["effective"] = effective_configuration(config, run.manifest["run_type"])
    check(fingerprint_inputs(proposed)["configuration"] == run.manifest["fingerprint_inputs"]["configuration"],
          "RUN_CONFIGURATION_CHANGED")


class Sources:
    """Verified source reconstruction; immutable copies contain no task workspaces."""
    def __init__(self, run, source):
        from run_metadata import source_snapshot, copy_source_snapshot
        self.run = run
        self.preserve_source = (run.manifest['run_type'] in ('wgm_v2', 'dgm') or
                                run.manifest.get('root_initialization', {}).get('comparison_import') is True)
        self.root = run.directory / "initial/src"
        expected = run.manifest["policy"]["initial_source_snapshot"]
        if self.root.exists():
            check(source_snapshot(self.root) == expected, "ROOT_SOURCE_CHANGED")
        else:
            self.root.parent.mkdir(exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".tmp-", dir=self.root.parent) as temporary:
                staged = Path(temporary) / "src"
                copy_source_snapshot(source, staged, expected)
                staged.rename(self.root)
                el._sync_dir(self.root.parent)

    def get(self, candidate):
        from candidate_provenance import read_chain, replay, source_digest
        from candidate_workspace import _copy_source, _git
        from downstream_solver import load_agent_policy
        chain = read_chain(self.run.directory, candidate, require_versioned=True)
        chain_id = [item.provenance.to_dict() for item in chain]
        if not chain:
            path = self.root
        else:
            folder = self.run.directory / "evaluation_sources"
            el._mkdir(folder)
            path = folder / candidate
            if not path.exists():
                with tempfile.TemporaryDirectory(prefix=".tmp-", dir=folder) as temporary:
                    staged = Path(temporary) / "agent"
                    _copy_source(self.root, staged, False,
                                 preserve_source=self.preserve_source)
                    _git(staged, "init", "-q", "--template=")
                    replay(staged, chain)
                    staged.rename(path)
                    el._sync_dir(folder)
            check(source_digest(path) == chain[-1].provenance.candidate_source_sha256, "CANDIDATE_SOURCE_CHANGED")
        policy = load_agent_policy(path)
        return {"path": path, "policy": policy, "policy_fingerprint": hashlib.sha256(policy.encode()).hexdigest(),
                "source_fingerprint": source_digest(path), "chain_digest": el.digest(chain_id)}


class ScopedRecorder:
    def __init__(self, parent, plan, execution_spec=None):
        self.parent, self.plan = parent, plan
        self.spec = execution_spec or plan.spec()
        self.call_phase = "final_evaluation" if plan.identity.stage == "final_benchmark" else "downstream"
        self.task_id, self.selection_digest = plan.identity.task_id, plan.selection_digest

    def record_call(self, row):
        check(row["call_id"] == self.spec.model_call_id, "USAGE_CALL_IDENTITY")
        self.parent.record_call({**row, "evaluation_id": self.plan.identity.evaluation_id,
                                 "evaluation_stage": self.plan.identity.stage})

    def refresh(self):
        return self.parent.refresh()

    def record_readiness(self, row):
        return self.parent.record_readiness(row)


def executor(config, run, entries, sources, recorder, *, cancel_event=None):
    from run_metadata import ROOT, file_hash
    check(model_fingerprint(config) == run.manifest["evaluation_execution"]["model_fingerprint"], "MODEL_FINGERPRINT_CHANGED")
    from evaluation_plan import evaluator_fingerprint
    check(evaluator_fingerprint(config.evaluation.polyglot) == run.manifest["evaluation_execution"]["evaluator_fingerprint"],
          "EVALUATOR_FINGERPRINT_CHANGED")
    def validate(plan):
        source = sources.get(plan.identity.candidate_id)
        entry = entries[plan.identity.task_id]
        check(el.digest(entry) == plan.identity.task_input_fingerprint and entry["base_commit"] == plan.base_commit,
              "TASK_INPUT_CHANGED")
        for name in ("source_fingerprint", "policy_fingerprint"):
            check(source[name] == getattr(plan.identity, name), "CANDIDATE_FINGERPRINT_CHANGED")
        return source, entry
    for planned in run.plans.read():
        validate(planned)  # Validate the entire saved schedule before the first callback.

    def execute(plan, ledger, *, execution_spec=None):
        if config.evaluation.polyglot:
            from polyglot_lifecycle import process_entry
        else:
            from swe_bench.harness import process_entry
        spec = execution_spec or plan.spec()
        check(spec.identity == plan.identity and spec.base_commit == plan.base_commit, 'EXECUTION_SPEC_CHANGED')
        source, entry = validate(plan)
        return process_entry(entry, ledger._folder(plan.identity.evaluation_id), plan.identity.candidate_id,
            backend="codex_exec", codex_config=(replace(config.codex, readiness_cancel_event=cancel_event)
                if cancel_event is not None else config.codex), evaluation_timeout=config.execution.evaluation_timeout,
            policy=source["policy"], source_fingerprint=source["source_fingerprint"],
            recorder=ScopedRecorder(recorder, plan, spec), execution_spec=spec, evaluation_ledger=ledger)
    return execute


def fixed_tasks(run, entries, candidate, source, execute, *, stage, tasks, selection_digest=None,
                workers=None, cancel_event=None):
    """Frozen canonical order; failed/ambiguous earlier ordinals stop the batch."""
    existing = [p for p in run.plans.read() if p.identity.stage == stage]
    check(len(existing) <= len(tasks), "FIXED_PLAN_COUNT")
    for index, task in enumerate(tasks):
        if index < len(existing):
            plan = existing[index]
            expected = run.plan(entries[task], stage=stage, candidate_id=candidate,
                policy_fingerprint=source["policy_fingerprint"], source_fingerprint=source["source_fingerprint"],
                ordinal=plan.identity.ordinal, selection_digest=selection_digest)
            check(plan == expected, "FIXED_PLAN_CHANGED")
        else:
            plan = run.plan(entries[task], stage=stage, candidate_id=candidate,
                policy_fingerprint=source["policy_fingerprint"], source_fingerprint=source["source_fingerprint"],
                selection_digest=selection_digest)
            run.publish(plan)
            existing.append(plan)
    results = []
    complete = False
    try:
        if workers is None:  # Existing HGM root/final callers retain the original serial scheduler.
            run.precheck()
            for plan in existing:
                value = run.execute(plan, execute)
                if value.outcome != 'success':
                    raise SchedulingBlocked('BOUND_FAILURE', plan)
                results.append(value)
        else:
            from evaluation_batch import execute_batch
            results = execute_batch(run, existing, execute, workers=workers, cancel_event=cancel_event)
        complete = True
    finally:
        # A read-only-derived summary is not a second source of execution state.
        results = []
        for plan in existing:
            snapshot = run.ledger.inspect(plan.identity.evaluation_id, plan.identity).snapshot
            if snapshot and snapshot.state == 'bound':
                result = read_binding(run.ledger, plan)
                if result.outcome == 'success':
                    results.append(result)
        from evaluation_batch import batch_reports
        summary = {"schema_version": 1, "execution_policy": "validated_observation_v2", "stage": stage,
            "status": "completed" if complete else "incomplete", "requested": len(tasks),
            "completed": len(results), "resolved": sum(r.resolved for r in results),
            "accuracy": sum(r.resolved for r in results) / len(tasks) if complete and tasks else None,
            "formal_accuracy": sum(r.resolved for r in results) / len(tasks)
                if complete and len(tasks) == pool_size(run.manifest["configuration"]["effective"]["evaluation"]["polyglot"]) else None,
            "selection_digest": selection_digest, "evaluation_parallelism": run.manifest.get('evaluation_parallelism'),
            "evaluation_batches": batch_reports(run.directory)}
        durable_json(run.directory / (stage + "_summary.json"), summary, replace_existing=True)
    return results


def refresh_views(run, *, candidates=None):
    """Rebuild disposable candidate/diagnosis views solely from validated bindings.

    Every view is under candidate/predictions/evaluation_id, so repeated task IDs
    cannot overwrite each other. Search never reads these views as observations.
    """
    from candidate_provenance import ROOT, read_chain
    by_candidate = defaultdict(list)
    for plan in run.plans.read():
        if candidates is not None and plan.identity.candidate_id not in candidates:
            continue
        if plan.identity.stage not in ("root", "search"):
            continue
        inspected = run.ledger.inspect(plan.identity.evaluation_id, plan.identity)
        if run.manifest['run_type'] == 'dgm':
            check(not inspected.error_code and not inspected.legacy and not inspected.read_only,
                  'VIEW_BINDING_INVALID')
        snap = inspected.snapshot
        if snap and snap.state == "bound":
            result = read_binding(run.ledger, plan)
            if result.outcome == "success":
                by_candidate[plan.identity.candidate_id].append(result)
    for candidate, results in by_candidate.items():
        chain = read_chain(run.directory, candidate, require_versioned=True)
        folder = run.directory / candidate
        folder.mkdir(exist_ok=True)
        metadata_path = folder / "metadata.json"
        metadata = el._read_json(metadata_path) if metadata_path.exists() else {
            "candidate_schema_version": 1, "provenance": ROOT.to_dict(), "run_id": "initial"}
        if candidate != "initial":
            check(bool(chain), "VIEW_CANDIDATE_PROVENANCE")
        tasks = list(dict.fromkeys(r.plan.identity.task_id for r in results))
        resolved = [task for task in tasks if next(r for r in results if r.plan.identity.task_id == task).resolved]
        metadata["downstream_backend"] = "codex_exec"
        metadata["overall_performance"] = {"accuracy_score": len(resolved)/len(tasks),
            "total_resolved_instances": len(resolved), "total_submitted_instances": len(tasks),
            "files": [], "total_unresolved_ids": [t for t in tasks if t not in resolved],
            "total_emptypatch_ids": [], "total_resolved_ids": resolved, "total_submitted_ids": tasks}
        for r in results:
            p = r.plan
            view = folder / "predictions" / p.identity.evaluation_id
            view.mkdir(parents=True, exist_ok=True)
            if run.manifest['run_type'] == 'dgm':
                # read_binding above verified the success, identity and sealed artifacts.
                # A waived historical attempt is never a business result source.
                from evaluation_adapter import EvaluationLifecycleAdapter
                inspected = run.ledger.inspect(p.identity.evaluation_id, p.identity)
                check(not inspected.error_code and inspected.snapshot is not None
                      and inspected.snapshot.state == 'bound'
                      and inspected.snapshot.binding_id == r.binding_digest, 'VIEW_BINDING_CHANGED')
                bound = next(el._decode(e.data) for e in reversed(inspected.snapshot.events) if e.kind == 'bound')
                check(bound['outcome'] == 'success' and any(a.attempt_id == bound['attempt_id']
                      and a.state == 'bound' for a in inspected.snapshot.attempts), 'VIEW_BINDING_ATTEMPT')
                spec = p.spec(bound['attempt_id'])
                patch = EvaluationLifecycleAdapter(run.ledger, None)._sealed(spec).patch.decode('utf-8')
            else:
                # The bound attempt (an audited retry binds a later attempt than attempt-000001).
                inspected = run.ledger.inspect(p.identity.evaluation_id, p.identity)
                check(not inspected.error_code and inspected.snapshot is not None, 'VIEW_BINDING_CHANGED')
                attempt = next(el._decode(e.data) for e in reversed(inspected.snapshot.events)
                               if e.kind == 'bound')['attempt_id']
                artifact = run.ledger._folder(p.identity.evaluation_id) / "artifacts" / attempt
                patch = el._regular(artifact / "patch.diff").read_text()
            prediction = {"instance_id": p.identity.task_id, "evaluation_id": p.identity.evaluation_id,
                "binding_digest": r.binding_digest, "model_patch": patch, "success": True,
                "eval_result": "resolved" if r.resolved else "unresolved", "derived_view": True}
            durable_json(view / (p.identity.task_id + ".json"), prediction, replace_existing=True)
            # Only safe summaries; raw prompts/stdout/stderr were never sealed.
            for suffix in (".md", "_eval.md"):
                path = view / (p.identity.task_id + suffix)
                from candidate_workspace import _atomic_write
                _atomic_write(path, ("Validated task outcome: " + prediction["eval_result"] + "\n").encode())
        durable_json(metadata_path, metadata, replace_existing=True)


def run_baseline(config, out_dir, *, n_tasks, instance_ids, agent_path, resume,
                 run_type, require_pinned_model=False, command=None):
    from run_metadata import build_manifest, RunRecorder, ROOT
    from task_schedule import canonical_tasks, unique_tasks
    require_supported(config)
    tasks = unique_tasks(instance_ids)[:n_tasks] if instance_ids is not None else canonical_tasks(ROOT, config.evaluation.task_order_seed, polyglot=config.evaluation.polyglot)[:n_tasks]
    check(type(n_tasks) is int and n_tasks > 0 and len(tasks) == n_tasks, "BASELINE_TASK_COUNT")
    source = Path(agent_path or ROOT)
    proposed = build_manifest(config, out_dir, run_type, tasks, source=source,
        require_pinned_model=require_pinned_model, command=command,
        shuffle=instance_ids is None, seed=config.evaluation.task_order_seed)
    run = _open(config, out_dir, proposed, resume)
    with run.lock():
        check(not (run.directory / "search_state.json").exists(), "BASELINE_SEARCH_STATE_UNEXPECTED")
        entries, sources = _entry_map(config, tasks), Sources(run, source)
        recorder = RunRecorder(run.directory, run.manifest, network_readiness=config.network_readiness)
        cancel = threading.Event()
        try:
            execute = executor(config, run, entries, sources, recorder, cancel_event=cancel)
            fixed_tasks(run, entries, "initial", sources.get("initial"), execute,
                        stage="root", tasks=tasks, workers=config.execution.evaluation_workers, cancel_event=cancel)
            refresh_views(run)
        except BaseException:
            recorder.refresh("incomplete")
            raise
        recorder.refresh("completed")
    return run.directory


def freeze_selection(run, nodes, sources, requested_tasks=None):
    """Resume reads the frozen selection without calling best_belief again."""
    tasks = run.manifest["tasks"]["ordered_ids"]
    requested_tasks = tasks if requested_tasks is None else list(requested_tasks)
    check(bool(requested_tasks) and requested_tasks == tasks[:len(requested_tasks)], "FINAL_TASK_ORDER")
    path = run.directory / "validated_final_selection.json"
    if path.exists():
        row = el._read_json(path)
        check(row["sha256"] == el.digest(row["selection"]), "FINAL_SELECTION_DIGEST")
        selection = row["selection"]
        check(selection["run_fingerprint"] == run.manifest["reproducibility_fingerprint"], "FINAL_SELECTION_IDENTITY")
        check(selection["task_ids"] == tasks and selection.get("requested_task_ids") == requested_tasks,
              "FINAL_TASK_ORDER_CHANGED")
    else:
        from search_state import best_belief
        chosen = best_belief(nodes)
        check(chosen["selected"] is not None, "FINAL_NO_CANDIDATE")
        source = sources.get(chosen["selected"]["candidate_id"])
        selection = {"run_fingerprint": run.manifest["reproducibility_fingerprint"], "best_belief": chosen,
            "task_ids": tasks, "requested_task_ids": requested_tasks,
            **{k: source[k] for k in ("source_fingerprint", "policy_fingerprint", "chain_digest")}}
        row = {"selection": selection, "sha256": el.digest(selection)}
        durable_json(path, row)
    source = sources.get(selection["best_belief"]["selected"]["candidate_id"])
    check(all(source[k] == selection[k] for k in ("source_fingerprint", "policy_fingerprint", "chain_digest")),
          "FINAL_SOURCE_CHANGED")
    return row, source


def _final_locked(config, run, nodes, sources, entries, recorder):
    tasks = run.manifest["tasks"]["ordered_ids"]
    limit = config.protocol.final_eval_task_limit or len(tasks)
    from task_schedule import pool_size
    check(limit <= len(tasks) and (config.protocol.effective_name == "smoke"
                                   or len(tasks) == limit == pool_size(config.evaluation.polyglot)), "FINAL_TASK_COUNT")
    selection, source = freeze_selection(run, nodes, sources, tasks[:limit])
    fixed_tasks(run, entries, selection["selection"]["best_belief"]["selected"]["candidate_id"], source,
        executor(config, run, entries, sources, recorder), stage="final_benchmark", tasks=tasks[:limit],
        selection_digest=selection["sha256"])
    return el._read_json(run.directory / "final_benchmark_summary.json")


def run_final(config, run_dir, nodes, source, recorder):
    require_supported(config)
    check_resume(run_dir, config)
    run = ExecutionRun(run_dir, el._read_json(Path(run_dir) / "run_manifest.json"))
    with run.lock():
        _check_configuration(config, run)
        _check_search_checkpoint(run)
        return _final_locked(config, run, nodes, Sources(run, source),
                             _entry_map(config, run.manifest["tasks"]["ordered_ids"]), recorder)


def run_search(config, *, preflight):
    if config.search.algorithm == "wgm_v2":
        from wgm_v2_search import run_search as run_v2
        return run_v2(config, preflight=preflight)
    import numpy as np
    import hgm_utils
    from tree import Node
    from run_metadata import ROOT, build_manifest, RunRecorder
    from task_schedule import canonical_tasks
    from search_config import is_corrected
    from search_state import SearchState, load_checkpoint
    from validated_search import ValidatedSearch
    from candidate_provenance import require_candidate, require_candidate_backend, validate_search_nodes
    from search_operations import SearchOperations
    from root_cache import root_reference
    from hgm_incremental_config import enabled as incremental_hgm
    if incremental_hgm(config):
        from protocol_config import configure_search_protocol
        configure_search_protocol(config)
        check(bool(config.paths.root_cache_dir), "HGM_INCREMENTAL_VERIFIED_ROOT_REQUIRED")
    require_supported(config)
    require_candidate_backend(config.llm.self_improve_backend)
    check(config.llm.diagnose_backend == "codex_exec", "VALIDATED_BACKEND_UNSUPPORTED")
    check(not config.paths.initial_baseline_dir, "VALIDATED_ROOT_IMPORT_NOT_IMPLEMENTED")
    cached = None
    if config.paths.root_cache_dir:
        from comparison_root import hgm_root_input
        source, cached = hgm_root_input(config)
    directory = config.paths.continue_from or config.paths.output_dir
    check(directory is not None, "VALIDATED_EXPLICIT_OUTPUT_REQUIRED")
    if config.paths.continue_from:
        check_resume(directory, config)
        check(not config.paths.output_dir or Path(config.paths.output_dir).absolute() == Path(directory).absolute(), "RESUME_DIRECTORY")
    elif Path(directory).exists():
        raise SchedulingBlocked("RUN_DIRECTORY_EXISTS")
    if cached is None:
        source = ROOT
    if cached is None and config.paths.initial_agent_name:
        candidate = ROOT / ("initial_polyglot" if config.evaluation.polyglot else "initial_swe") / config.paths.initial_agent_name / "src"
        check(candidate.is_dir(), "INITIAL_SOURCE_MISSING")
        source = candidate
    tasks = canonical_tasks(ROOT, config.evaluation.task_order_seed if is_corrected(config) else 42,
                            polyglot=config.evaluation.polyglot)
    proposed = build_manifest(config, directory, config.search.algorithm, tasks, source=source,
                              require_clean=False, shuffle=True,
                              seed=config.evaluation.task_order_seed if is_corrected(config) else 42,
                              root_initialization=(dict(root_reference(cached),
                                  comparison_import=True, search_budget_charge=0,
                                  comparison_protocol=config.protocol.effective_name,
                                  observation_source='historical_verified_root') if cached else None))
    if cached:
        from comparison_root import common_root
        verified_root = common_root(proposed, config.paths.root_cache_dir)
        if incremental_hgm(config):
            from run_metadata import fingerprint_inputs, digest
            certificate, _, _ = verified_root
            proposed["hgm_incremental_inputs"] = {"common_root": certificate}
            proposed["fingerprint_inputs"] = fingerprint_inputs(proposed)
            proposed["reproducibility_fingerprint"] = digest(proposed["fingerprint_inputs"])
    # Identity validation is before preflight as well as every execution boundary.
    if config.paths.continue_from:
        saved = el._read_json(Path(directory) / "run_manifest.json")
        check(saved["reproducibility_fingerprint"] == proposed["reproducibility_fingerprint"], "RUN_RESUME_FINGERPRINT")
        check((Path(directory) / "evaluation_ledger/run.json").is_file(), "LEGACY_WITHOUT_EVALUATION_LEDGER")
        if incremental_hgm(config):
            from hgm_incremental import read_control
            read_control(directory, config, saved)
    preflight(config, output_root=directory)
    run = _open(config, directory, proposed, bool(config.paths.continue_from))
    with run.lock():
        _check_search_checkpoint(run)
        if (run.directory / "search_state.json").exists():
            from search_operations import read_checkpoint
            read_checkpoint(run.directory)  # Reject legacy/invalid RNG before root callbacks.
        entries, sources = _entry_map(config, tasks), Sources(run, source)
        if cached and not (run.directory / 'initial/metadata.json').exists():
            import shutil
            from candidate_workspace import _atomic_write
            shutil.copytree(source.parent / 'predictions', run.directory / 'initial/predictions', dirs_exist_ok=True)
            # Metadata is the last publication, so an interrupted copy is locally repeatable.
            _atomic_write(run.directory / 'initial/metadata.json', (source.parent / 'metadata.json').read_bytes())
        recorder = RunRecorder(run.directory, run.manifest, network_readiness=config.network_readiness)
        execute = executor(config, run, entries, sources, recorder)
        try:
            if cached:
                root_results = {r['task_id']: int(r['result'] == 'resolved') for r in cached['per_task_results']}
            else:
                roots = fixed_tasks(run, entries, "initial", sources.get("initial"), execute, stage="root", tasks=tasks)
                root_results = {r.plan.identity.task_id: int(r.resolved) for r in roots}
            refresh_views(run)
            hgm_utils.nodes.clear()
            hgm_utils.init(config.evaluation.polyglot, str(run.directory), tasks, 0, config.llm.self_improve_llm, config.execution.self_improve_timeout)
            root = Node("initial", [root_results[t] for t in tasks])
            root.completed_task_results = root_results
            submitted = defaultdict(set, {0: set(tasks)})
            state = SearchState(config, hgm_utils.nodes, submitted, tasks)
            def make_plan(node, task, draw):
                src = sources.get(node.commit_id)
                return run.plan(entries[task], stage="search", candidate_id=node.commit_id,
                    task_draw_index=draw, policy_fingerprint=src["policy_fingerprint"], source_fingerprint=src["source_fingerprint"])
            engine = ValidatedSearch(run, state, Node, make_plan=make_plan, execute=execute,
                generate=None, operations=SearchOperations(run, config, sources, entries, recorder),
                refresh_views=lambda: refresh_views(run))
            if (run.directory / "search_state.json").exists():
                saved = load_checkpoint(run.directory)["state"]
                pending = saved["history"][-1] if saved["inflight"] and saved["inflight"]["kind"] == "expansion" else None
                validate_search_nodes(run.directory, saved["nodes"], pending)
                engine.restore()
            else:
                random.seed(config.search.seed if is_corrected(config) else 42)
                if not is_corrected(config):
                    # Match the original task shuffle's global RNG consumption.
                    original = json.loads((ROOT / "swe_bench/subsets/small.json").read_text()) + json.loads((ROOT / "swe_bench/subsets/medium.json").read_text())
                    random.shuffle(original)
                np.random.seed(config.search.seed)
                engine.save()
            if incremental_hgm(config):
                import time
                from hgm_incremental import IncrementalHGM
                IncrementalHGM(engine, root_results, wall_seconds=lambda:
                    recorder.previous_total + time.monotonic() - recorder.started).run_to_target()
            else:
                engine.run_to_budget()
            hgm_utils.n_task_evals = engine.count
            if is_corrected(config):
                _final_locked(config, run, list(state.nodes.values()), sources, entries, recorder)
            recorder.refresh("completed")
        except BaseException:
            recorder.refresh("incomplete")
            raise
    return run.directory
