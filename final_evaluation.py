"""One frozen selected agent, evaluated independently of all search state."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import time

from candidate_workspace import _copy_source, _git, _check_workspace
from candidate_provenance import read_chain, replay
from run_metadata import atomic_json, digest, file_hash, source_snapshot
from root_cache import validate_task_result
from search_config import is_corrected
from search_state import best_belief


def patch_chain(run_dir, candidate_id):
    """Compatibility API returning the verified COMPLETE replay chain."""
    return _chain_rows(read_chain(run_dir, candidate_id))


def _chain_rows(chain):
    return [{"candidate_id": item.provenance.candidate_id,
             "patch_sha256": item.provenance.self_patch.patch_sha256,
             "metadata_sha256": item.metadata_sha256} for item in chain]


class FinalRecorder:
    call_phase = "final_evaluation"

    def __init__(self, parent, directory, task_id, selection_digest):
        self.parent, self.directory, self.task_id = parent, Path(directory), task_id
        self.selection_digest = selection_digest

    def record_call(self, record):
        if record["phase"] != self.call_phase or record.get("instance_id") != self.task_id:
            raise ValueError("Final invocation phase/task mismatch")
        directory = self.directory / "codex_calls"
        directory.mkdir(exist_ok=True)
        for path in directory.glob("*.json"):
            saved = json.loads(path.read_text())
            if saved.get("instance_id") == self.task_id and saved["call_id"] != record["call_id"]:
                raise ValueError("Final task already has an invocation; refusing a duplicate")
        atomic_json(directory / (record["call_id"] + ".json"), record)
        self.parent.record_call(record)

    def refresh(self):
        self.parent.refresh()

    def record_readiness(self, row):
        self.parent.record_readiness(row)


def evaluate_one(config, task_id, predictions, source, recorder):
    """Use the same host solver and evaluator, without hgm_utils.eval_agent."""
    from swe_bench.harness import harness
    return harness(test_task_list=[task_id], n_tasks=1, max_workers=config.execution.max_workers,
                   model_name_or_path="selected-final-agent", init_agent_path=str(source),
                   pred_dname=str(predictions.parent), resume_dir=predictions,
                   backend=config.llm.downstream_backend, codex_config=config.codex,
                   evaluation_timeout=config.execution.evaluation_timeout,
                   dataset_name=config.evaluation.dataset_name, dataset_path=config.evaluation.dataset_path,
                   recorder=recorder)


def final_summary(directory, selection, progress):
    from runners.usage import aggregate_usage
    from collections import Counter
    directory = Path(directory)
    outcomes = list(progress["results"].values())
    resolved = sum(r["result"] == "resolved" for r in outcomes)
    completed = sum(r["result"] in ("resolved", "unresolved") for r in outcomes)
    protocol = selection["identity"].get("protocol", "legacy/unavailable")
    full = (protocol in ("formal", "pilot") and len(selection["identity"]["ordered_task_ids"]) == 60
            and len(selection["identity"]["requested_task_ids"]) == 60 and completed == 60)
    calls = [json.loads(p.read_text()) for p in (directory / "codex_calls").glob("*.json")]
    return {"protocol": protocol, "search_budget": selection["identity"].get("search_budget"),
            "selected": selection["identity"]["selected"], "selection_digest": selection["selection_digest"],
            "requested_tasks": len(selection["identity"]["requested_task_ids"]),
            "submitted_instances": len(progress["attempted_tasks"]), "completed_instances": completed,
            "resolved_instances": resolved, "failed_instances": len(outcomes)-completed,
            "pending_instances": len(selection["identity"]["requested_task_ids"])-len(outcomes),
            "final_evaluation_complete": full, "final_accuracy": resolved/60 if full else None,
            "status": ("completed" if completed == len(selection["identity"]["requested_task_ids"]) else
                       "failed" if len(outcomes) == len(selection["identity"]["requested_task_ids"]) else "interrupted"),
            "duration_seconds": progress["duration_seconds"], "codex_usage": aggregate_usage(calls)["final_evaluation"],
            "failure_stage_counts": dict(Counter(r.get("failure_stage", "unavailable") for r in outcomes
                                                 if r["result"] not in ("resolved", "unresolved"))),
            "failure_results": {r["task_id"]:r["result"] for r in outcomes if r["result"] not in ("resolved", "unresolved")}}


def run_final_evaluation(config, run_dir, nodes, source, recorder, *, evaluator=None):
    from execution_policy import validated, check_resume
    if (Path(run_dir) / "run_manifest.json").is_file():
        check_resume(run_dir, config)
    if validated(config):
        from validated_runs import run_final
        return run_final(config, run_dir, nodes, source, recorder)
    started = time.monotonic()
    if not is_corrected(config):
        raise ValueError("Legacy HGM does not automatically select or evaluate a final agent")
    if config.evaluation.polyglot:
        raise ValueError("Final evaluation protocol currently requires SWE")
    config.protocol.validate()
    from task_schedule import validate_schedule
    validate_schedule(recorder.manifest["tasks"])
    run_dir, source = Path(run_dir), Path(source)
    report = best_belief(nodes)
    selected = report["selected"]
    if selected is None:
        raise ValueError("Final selection has no eligible node")
    task_ids = recorder.manifest["tasks"]["ordered_ids"]
    limit = config.protocol.final_eval_task_limit or len(task_ids)
    if limit > len(task_ids) or (config.protocol.effective_name != "smoke" and len(task_ids) != 60):
        raise ValueError("Invalid final evaluation task count")
    expected_source = recorder.manifest["policy"]["initial_source_snapshot"]
    if source_snapshot(source) != expected_source:
        raise ValueError("Final root source changed")
    verified_chain = read_chain(run_dir, selected["candidate_id"])
    identity = {"schema_version": 1, "search_fingerprint": recorder.manifest["reproducibility_fingerprint"],
                "protocol": config.protocol.effective_name,
                "search_budget": config.protocol.search_budget(config.execution.max_task_evals),
                "selected": deepcopy(selected), "method": {k:report[k] for k in ("method", "epsilon_percent", "q", "convention")},
                # Historical final-selection wire key means complete replay;
                # it is never written into candidate ancestor-only metadata.
                "ancestor_chain": _chain_rows(verified_chain),
                "root_source": expected_source, "ordered_task_ids": list(task_ids),
                "requested_task_ids": list(task_ids[:limit]),
                "execution_config": {"codex": asdict(config.codex), "execution": asdict(config.execution),
                                     "evaluation": asdict(config.evaluation),
                                     "protocol": config.protocol.identity(config.execution.max_task_evals)},
                "configuration": deepcopy(recorder.manifest["configuration"]["effective"])}
    directory = run_dir / "final_evaluation"
    frozen = directory / "frozen"
    selection_path = frozen / "selection.json"
    if selection_path.exists():
        selection = json.loads(selection_path.read_text())
        body = {k:v for k,v in selection.items() if k != "selection_digest"}
        if selection["identity"] != identity or selection["selection_digest"] != digest(body):
            raise ValueError("Final selection, patch chain or configuration changed; refusing resume")
    else:
        directory.mkdir(exist_ok=True)
        # An interrupted reconstruction publishes neither agent nor selection.
        # Any already-published unbound agent is ambiguous, never overwritten.
        if frozen.exists():
            raise ValueError("Unbound final agent exists; refusing to overwrite it")
        with tempfile.TemporaryDirectory(prefix=".candidate-final-", dir=directory) as temp:
            staging = Path(temp) / "frozen"
            staging.mkdir()
            workspace = staging / "agent"
            _copy_source(source, workspace, False)
            _git(workspace, "init", "-q", "--template=")
            (staging / "patches").mkdir()
            for index, item in enumerate(identity["ancestor_chain"]):
                data = (run_dir / item["candidate_id"] / "model_patch.diff").read_bytes()
                import hashlib
                if hashlib.sha256(data).hexdigest() != item["patch_sha256"]:
                    raise ValueError("Final patch changed during reconstruction")
                (staging / "patches" / f"{index:06d}.diff").write_bytes(data)
            replay(workspace, verified_chain)
            _check_workspace(workspace)
            selection = {"identity": identity, "agent_source": source_snapshot(workspace),
                         "agent_policy_sha256": file_hash(workspace / "prompts/downstream_instructions.txt")}
            selection["selection_digest"] = digest(selection)
            # Publish selection, patches and reconstructed agent in one rename.
            atomic_json(staging / "selection.json", selection, exclusive=True)
            staging.rename(frozen)
    for index, item in enumerate(identity["ancestor_chain"]):
        if file_hash(frozen / "patches" / f"{index:06d}.diff") != item["patch_sha256"]:
            raise ValueError("Frozen final patch changed")
    if (source_snapshot(frozen / "agent") != selection["agent_source"] or
            file_hash(frozen / "agent/prompts/downstream_instructions.txt") != selection["agent_policy_sha256"]):
        raise ValueError("Frozen final agent content changed")
    best_path = run_dir / "best_belief.json"
    if best_path.exists() and json.loads(best_path.read_text()) != report:
        raise ValueError("Frozen best-belief report changed")
    if not best_path.exists():
        atomic_json(best_path, report, exclusive=True)
    progress_path = directory / "progress.json"
    progress = (json.loads(progress_path.read_text()) if progress_path.exists() else
                {"selection_digest": selection["selection_digest"], "attempted_tasks": [],
                 "results": {}, "duration_seconds": 0.0})
    if progress["selection_digest"] != selection["selection_digest"]:
        raise ValueError("Final progress belongs to another selection")
    if (progress["attempted_tasks"] != identity["requested_task_ids"][:len(progress["attempted_tasks"])] or
            set(progress["results"]) - set(progress["attempted_tasks"])):
        raise ValueError("Invalid final task progress")
    prior_duration = progress["duration_seconds"]
    def persist():
        progress["duration_seconds"] = prior_duration + time.monotonic()-started
        atomic_json(progress_path, progress)
        summary = final_summary(directory, selection, progress)
        atomic_json(directory / "summary.json", summary)
        return summary
    def result(task):
        return validate_task_result(directory / "predictions", task, directory / "codex_calls",
            phase="final_evaluation", policy_hash=selection["agent_policy_sha256"],
            model=config.codex.model, reasoning=config.codex.reasoning_effort, require_complete=False,
            allow_prelaunch_failure=True)
    # Check all persisted results and invocation identities before any new task.
    for task, facts in progress["results"].items():
        if result(task) != facts:
            raise ValueError("Final completed task content changed")
    calls = [json.loads(p.read_text()) for p in (directory / "codex_calls").glob("*.json")]
    call_tasks = [c.get("instance_id") for c in calls]
    if (len(set(call_tasks)) != len(call_tasks) or set(call_tasks) - set(progress["attempted_tasks"]) or
            any(c.get("phase") != "final_evaluation" or c.get("final_selection_digest") != selection["selection_digest"] for c in calls)):
        raise ValueError("Final invocation identity mismatch")
    for call in calls:
        task = call["instance_id"]
        if not (directory / "predictions" / (task + ".json")).is_file():
            continue
        result(task)  # A terminal result binds the authoritative scoped ledger.
        parent_path = recorder.directory / "codex_calls" / (call["call_id"] + ".json")
        previous = json.loads(parent_path.read_text()) if parent_path.is_file() else None
        if previous != call:
            identity_keys = ("call_id", "phase", "instance_id", "final_selection_digest", "configured_model", "configured_reasoning_effort")
            if previous is not None and (any(previous.get(k) != call.get(k) for k in identity_keys)
                                         or previous.get("status") not in ("attempting", "running", "interrupted")):
                raise ValueError("Parent and final call ledger identities disagree")
            # Recover a crash between scoped-ledger and parent-ledger persistence;
            # this overwrites the same call ID and cannot launch another call.
            recorder.record_call(call)
    try:
        for task in identity["requested_task_ids"]:
            prediction = directory / "predictions" / (task + ".json")
            if prediction.is_file():
                facts = result(task)
                if task in progress["results"] and progress["results"][task] != facts:
                    raise ValueError("Final completed task content changed")
                progress["results"][task] = facts
                if task not in progress["attempted_tasks"]:
                    raise ValueError("Unclaimed final task result")
                persist()
                continue
            calls = [json.loads(p.read_text()) for p in (directory / "codex_calls").glob("*.json")]
            if task in progress["results"] or any(r.get("instance_id") == task for r in calls):
                # The external CLI has no idempotency key. A crash after invocation
                # without a terminal result must never cause another invocation.
                raise ValueError("Ambiguous final task invocation; refusing automatic retry")
            if task not in progress["attempted_tasks"]:
                progress["attempted_tasks"].append(task)
            persist()
            scoped = FinalRecorder(recorder, directory, task, selection["selection_digest"])
            (evaluator or evaluate_one)(config, task, directory / "predictions", frozen / "agent", scoped)
            progress["results"][task] = result(task)
            persist()
        return persist()
    finally:
        persist()
        recorder.refresh()
