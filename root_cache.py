"""Verified baseline import. No runner, Docker or benchmark execution here."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

from run_metadata import (atomic_json, digest, file_hash, fingerprint_inputs, source_snapshot,
                          task_records, read_report)
from task_schedule import validate_schedule, CANONICAL_MEMBERSHIP_HASH, benchmark


def content_inventory(directory, *, exclude=()):
    directory = Path(directory)
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("Symlinks are not allowed in sealed artifacts")
        if path.is_file():
            name = path.relative_to(directory).as_posix()
            if name not in exclude:
                files[name] = {"sha256": file_hash(path), "executable": bool(path.stat().st_mode & 0o111)}
    return files


def seal_baseline(directory):
    directory = Path(directory)
    files = content_inventory(directory, exclude=("baseline_seal.json",))
    atomic_json(directory / "baseline_seal.json", {"schema_version": 1, "files": files,
                                                   "content_digest": digest(files)}, exclusive=True)


def verify_baseline_seal(directory):
    directory = Path(directory)
    path = directory / "baseline_seal.json"
    if not path.is_file():
        raise ValueError("Baseline has no content seal; legacy/unavailable")
    seal = json.loads(path.read_text())
    actual = content_inventory(directory, exclude=("baseline_seal.json",))
    if seal.get("schema_version") != 1 or seal.get("files") != actual or seal.get("content_digest") != digest(actual):
        raise ValueError("Baseline content digest changed")
    return seal


def agent_condition(manifest_codex):
    """Provider identity of a root; empty for the original Codex condition.

    Codex roots keep their historical keys. A Claude root also binds the
    provider and the digest of its provider-specific prompt texts.
    """
    provider = manifest_codex.get("provider")
    if provider is None:
        return {}
    return {"provider": provider, "prompt_identity": manifest_codex["prompt_identity"]}


def config_condition(codex_config):
    """The same identity computed from a run's configuration."""
    provider = getattr(codex_config, "provider", None)
    if provider is None:
        return {}
    from provider_prompts import identity
    return {"provider": provider, "prompt_identity": identity(provider)}


def same_condition(shared, codex_config):
    expected = config_condition(codex_config)
    return ({k: shared.get(k) for k in ("provider", "prompt_identity")}
            == {k: expected.get(k) for k in ("provider", "prompt_identity")})


def compatibility(manifest):
    """Only shared scientific inputs, never search algorithm or output paths."""
    tasks = manifest["tasks"]
    validate_schedule(tasks)
    cfg = manifest["configuration"]["effective"]
    model = manifest["codex"]["configured_model"]
    effort = manifest["codex"]["configured_reasoning_effort"]
    if not model or not effort:
        raise ValueError("Verified root requires explicit model and reasoning effort")
    if cfg["llm"]["downstream_backend"] != "codex_exec":
        raise ValueError("Verified root requires codex_exec")
    if manifest["repository"].get("tracked_clean") is not True or manifest["repository"].get("untracked_paths"):
        raise ValueError("Verified root requires a clean committed source")
    frozen = benchmark(cfg["evaluation"]["polyglot"])
    if len(tasks["ordered_ids"]) != frozen["count"]:
        raise ValueError("Verified root protocol requires the benchmark's canonical task pool")
    if tasks["task_membership_hash"] != frozen["membership_hash"]:
        raise ValueError("Verified root task membership differs from the frozen canonical set")
    evaluator = manifest.get("evaluator")
    if not evaluator or not evaluator.get("sha256"):
        raise ValueError("Evaluator fingerprint is unavailable")
    return {"creator_commit": manifest["repository"]["commit"],
            "backend": "codex_exec", "model": model, "reasoning_effort": effort,
            "ordered_task_ids": tasks["ordered_ids"],
            **{k: tasks[k] for k in ("task_membership_hash", "task_order_hash", "task_order_seed")},
            "dataset": manifest["dataset"], "subset_sha256": tasks["subset_sha256"],
            "evaluator": evaluator, "source": manifest["policy"]["initial_source_snapshot"],
            "policy_sha256": manifest["policy"]["downstream_instructions_sha256"],
            "runtime_prompt_hashes": manifest["policy"]["prompt_files_sha256"],
            "configured_workers": cfg["execution"]["max_workers"],
            "effective_workers": cfg["effective"]["workers"],
            "evaluation_timeout": cfg["execution"]["evaluation_timeout"],
            "codex_timeout": cfg["codex"]["timeout_seconds"],
            "codex_cli_version": manifest["codex"]["codex_cli_version"],
            **agent_condition(manifest["codex"])}


def validate_task_result(predictions, task_id, ledger, *, phase="downstream", policy_hash=None,
                         model=None, reasoning=None, require_complete=True, allow_prelaunch_failure=False):
    """Bind one prediction, metadata, patch and a single persisted invocation."""
    predictions, ledger = Path(predictions), Path(ledger)
    prediction_path = predictions / (task_id + ".json")
    metadata_path = predictions / "tasks" / task_id / "metadata.json"
    prediction = json.loads(prediction_path.read_text())
    metadata = json.loads(metadata_path.read_text())
    kind = prediction["eval_result"]
    if kind not in ("resolved", "unresolved", "error", "incomplete", "empty_patch"):
        raise ValueError("Unknown task result category")
    if prediction.get("instance_id") != task_id or metadata.get("instance_id") != task_id:
        raise ValueError("Task identity mismatch")
    if bool(prediction.get("success")) != (kind in ("resolved", "unresolved")):
        raise ValueError("Task category/success mismatch")
    if prediction.get("backend") != "codex_exec" or metadata.get("backend") != "codex_exec":
        raise ValueError("Task backend mismatch")
    if (require_complete or kind in ("resolved", "unresolved")) and (kind not in ("resolved", "unresolved") or metadata.get("stage") != "complete"
                             or metadata.get("status") != "success" or not metadata.get("success")
                             or not prediction.get("success") or metadata.get("interrupted")):
        raise ValueError("Task is not a complete resolved/unresolved evaluation")
    records = metadata.get("codex_call_records", [])
    if (allow_prelaunch_failure and not records and metadata.get("codex_calls") == 0
            and kind in ("error", "incomplete") and not prediction.get("success")):
        return {"task_id": task_id, "result": kind, "call_id": None, "call_count": 0,
                "prediction_sha256": file_hash(prediction_path), "metadata_sha256": file_hash(metadata_path),
                "failure_stage": metadata.get("stage", "unavailable")}
    if len(records) != 1 or metadata.get("codex_calls") != 1:
        raise ValueError("Task needs exactly one traceable Codex call")
    call = records[0]
    cid = call.get("call_id", "")
    if not isinstance(cid, str) or not cid or any(c not in "0123456789abcdef" for c in cid):
        raise ValueError("Invalid call ID")
    saved = json.loads((ledger / (cid + ".json")).read_text())
    if call != saved or call.get("phase") != phase or call.get("attempted_calls") != 1:
        raise ValueError("Task call ledger mismatch")
    if phase == "final_evaluation" and call.get("instance_id") != task_id:
        raise ValueError("Final call belongs to another task")
    if (require_complete or kind in ("resolved", "unresolved")) and (call.get("successful_calls") != 1 or call.get("started_calls") != 1
                             or call.get("failed_calls") != 0 or call.get("timed_out_calls") != 0):
        raise ValueError("Task invocation was not successful")
    if model is not None and (call.get("configured_model") != model or call.get("configured_reasoning_effort") != reasoning):
        raise ValueError("Task model/reasoning mismatch")
    if policy_hash is not None and metadata.get("policy_sha256") != policy_hash:
        raise ValueError("Task policy mismatch")
    if kind in ("resolved", "unresolved"):
        patch = prediction.get("model_patch", "")
        import hashlib
        patch_hash = hashlib.sha256(patch.encode()).hexdigest()
        if not patch.strip() or metadata.get("patch_sha256") != patch_hash or metadata.get("patch_bytes") != len(patch.encode()):
            raise ValueError("Task patch digest mismatch or empty patch")
        if file_hash(metadata_path.parent / "task_patch.diff") != patch_hash:
            raise ValueError("Stored task patch digest mismatch")
        if metadata.get("evaluation", {}).get("resolved") != (kind == "resolved"):
            raise ValueError("Task evaluation and prediction disagree")
    return {"task_id": task_id, "result": kind, "call_id": cid,
            "call_count": 1, "failure_stage": metadata.get("stage") if kind not in ("resolved", "unresolved") else None,
            "prediction_sha256": file_hash(prediction_path), "metadata_sha256": file_hash(metadata_path),
            "call_sha256": file_hash(ledger / (cid + ".json")), "patch_sha256": metadata.get("patch_sha256")}


PRELAUNCH_STAGES = ("dataset_image_preparation", "task_workspace_extraction")


def prelaunch_retries(directory, tasks):
    """Audited pre-launch retries: {task: original submission time (ns)}.

    Written by the out-of-tree prelaunch_retry tool. Each archived attempt must
    be a failure before any model launch (zero calls, pre-launch stage) whose
    files still match their recorded digests. Only such a task may have been
    executed after later tasks; its canonical position is its first submission.
    """
    base = Path(directory) / "evidence" / "prelaunch-retries"
    original = {}
    if not base.exists():
        return original
    for folder in sorted(base.iterdir()):
        if folder.name not in tasks:
            raise ValueError("Pre-launch retry evidence for an unknown task")
        for attempt in sorted(folder.glob("attempt-*")):
            record = json.loads((attempt / "retry.json").read_text())
            prediction = json.loads((attempt / "prediction.json").read_text())
            metadata = json.loads((attempt / "task/metadata.json").read_text())
            if (record.get("schema_version") != 1 or record.get("instance_id") != folder.name
                    or file_hash(attempt / "prediction.json") != record.get("prediction_sha256")
                    or file_hash(attempt / "task/metadata.json") != record.get("metadata_sha256")
                    or prediction.get("instance_id") != folder.name or prediction.get("success")
                    or prediction.get("eval_result") not in ("error", "incomplete")
                    or metadata.get("instance_id") != folder.name or metadata.get("stage") not in PRELAUNCH_STAGES
                    or metadata.get("codex_calls") != 0 or metadata.get("codex_call_records")
                    or record.get("original_submitted_at_ns") != metadata.get("submitted_at_ns")):
                raise ValueError("Invalid pre-launch retry evidence: " + folder.name)
            value = record["original_submitted_at_ns"]
            original[folder.name] = min(value, original.get(folder.name, value))
    return original


def _commit_is_ancestor(ancestor, descendant):
    if ancestor == descendant:
        return True
    from run_metadata import ROOT
    return subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", ancestor, descendant],
                          capture_output=True).returncode == 0


def validate_baseline(directory, target):
    directory = Path(directory)
    seal = verify_baseline_seal(directory)
    manifest, summary, run = read_report(directory)
    if manifest is None or manifest.get("run_type") != "default-baseline" or summary["status"] != "completed":
        raise ValueError("Root source must be a completed default-baseline")
    if run.get("status") != "complete":
        raise ValueError("Baseline run status is not complete")
    if digest(fingerprint_inputs(manifest)) != manifest["reproducibility_fingerprint"] or fingerprint_inputs(manifest) != manifest["fingerprint_inputs"]:
        raise ValueError("Baseline fingerprint is inconsistent")
    expected, actual = compatibility(target), compatibility(manifest)
    for key in expected:
        if key == "creator_commit":
            # The baseline may be imported from a later commit; every scientific
            # input (prompts, evaluator, source, policy, model, provider, CLI) is
            # still compared field by field below.
            if not _commit_is_ancestor(actual[key], expected[key]):
                raise ValueError("Baseline commit is not an ancestor of the importing commit")
        elif actual[key] != expected[key]:
            raise ValueError("Baseline compatibility mismatch: " + key)
    count = len(actual["ordered_task_ids"])
    if summary["submitted_instances"] != count or summary["completed_instances"] != count:
        raise ValueError("Baseline must have every canonical task completed")
    if any(summary[k + "_instances"] for k in ("error", "incomplete", "empty_patch")):
        raise ValueError("Baseline has unsuccessful task executions")
    if summary["resolved_instances"] + summary["unresolved_instances"] != count:
        raise ValueError("Baseline resolved/unresolved count mismatch")
    if source_snapshot(directory / "src") != actual["source"] or file_hash(directory / "src/prompts/downstream_instructions.txt") != actual["policy_sha256"]:
        raise ValueError("Baseline initial source/policy mismatch")
    tasks = actual["ordered_task_ids"]
    if (run.get("backend") != "codex_exec" or run.get("task_ids") != tasks or
            run.get("policy_sha256") != actual["policy_sha256"]):
        raise ValueError("Baseline run record identity disagrees with manifest")
    saved_summary = json.loads((directory / "run_summary.json").read_text())
    for key in ("submitted_instances", "completed_instances", "resolved_instances", "unresolved_instances",
                "error_instances", "incomplete_instances", "empty_patch_instances"):
        if saved_summary.get(key) != summary[key]:
            raise ValueError("Baseline saved summary/task counts disagree: " + key)
    rows = task_records(directory)
    if len(rows) != len(tasks) or set(r["metadata"]["instance_id"] for r in rows) != set(tasks):
        raise ValueError("Baseline task membership/duplicates mismatch")
    retried = prelaunch_retries(directory, tasks)
    def submitted(row):
        return retried.get(row["metadata"]["instance_id"], row["metadata"]["submitted_at_ns"])
    if [r["metadata"]["instance_id"] for r in sorted(rows, key=submitted)] != tasks:
        raise ValueError("Baseline actual task order mismatch")
    results = [validate_task_result(directory / "predictions", t, directory / "codex_calls",
               policy_hash=actual["policy_sha256"], model=actual["model"], reasoning=actual["reasoning_effort"]) for t in tasks]
    ids = [r["call_id"] for r in results]
    if len(set(ids)) != len(tasks) or {p.stem for p in (directory / "codex_calls").glob("*.json")} != set(ids):
        raise ValueError("Baseline contains duplicate or extra model calls")
    return manifest, actual, results, seal


def import_baseline(directory, destination, target):
    """New cache only. Re-import validates an existing cache and never repairs it."""
    directory, destination = Path(directory).resolve(), Path(destination).resolve()
    manifest, identity, results, seal = validate_baseline(directory, target)
    if destination.exists():
        cached = verify_root_cache(destination, target)
        if cached["baseline_manifest_sha256"] != file_hash(directory / "run_manifest.json"):
            raise ValueError("Existing root cache belongs to another baseline")
        return cached
    if destination.is_relative_to(directory) or directory.is_relative_to(destination):
        raise ValueError("Root cache and baseline must be separate directories")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".root-import-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "cache"
        initial = staging / "initial"
        initial.mkdir(parents=True)
        shutil.copytree(directory / "src", initial / "src")
        shutil.copytree(directory / "predictions", initial / "predictions/baseline")
        evidence = staging / "evidence"
        evidence.mkdir()
        for name in ("run_manifest.json", "run_summary.json", "run.json", "baseline_seal.json"):
            shutil.copy2(directory / name, evidence / name)
        shutil.copytree(directory / "codex_calls", evidence / "codex_calls")
        if (directory / "evidence" / "prelaunch-retries").exists():
            # Cache "evidence/<path>" mirrors baseline "<path>" (see verify_root_cache).
            shutil.copytree(directory / "evidence" / "prelaunch-retries", evidence / "evidence" / "prelaunch-retries")
        resolved = [r["task_id"] for r in results if r["result"] == "resolved"]
        unresolved = [r["task_id"] for r in results if r["result"] == "unresolved"]
        atomic_json(initial / "metadata.json", {"run_id": "initial", "downstream_backend": "codex_exec",
            "ancestor_patch_sha256": [], "swe_dnames": ["baseline"], "overall_performance": {
                "total_submitted_instances": len(identity["ordered_task_ids"]), "total_resolved_instances": len(resolved),
                "accuracy_score": len(resolved)/len(identity["ordered_task_ids"]), "total_submitted_ids": identity["ordered_task_ids"],
                "total_resolved_ids": resolved, "total_unresolved_ids": unresolved,
                "total_emptypatch_ids": [], "files": ["baseline"]}})
        cached = {"schema_version": 1, "verification": "verified_root_cache",
                  "baseline_run_directory": str(directory), "baseline_manifest_sha256": file_hash(directory / "run_manifest.json"),
                  "baseline_fingerprint": manifest["reproducibility_fingerprint"],
                  "baseline_content_digest": seal["content_digest"], "compatibility": identity,
                  "per_task_results": results, "resolved": len(resolved), "unresolved": len(unresolved),
                  "files": content_inventory(staging)}
        cached["content_digest"] = digest(cached)
        cached["created_at_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(staging / "root_cache_manifest.json", cached)
        # Detect a source changed during import, before publishing any cache.
        if verify_baseline_seal(directory) != seal:
            raise ValueError("Baseline changed during import")
        verify_root_cache(staging, target)
        staging.rename(destination)
    return cached


def verify_root_cache(directory, target=None, expected_digest=None):
    directory = Path(directory)
    path = directory / "root_cache_manifest.json"
    if not path.is_file():
        raise ValueError("legacy_unverified_cache: verified root manifest is missing")
    cached = json.loads(path.read_text())
    body = {k:v for k,v in cached.items() if k not in ("content_digest", "created_at_utc")}
    if (cached.get("schema_version") != 1 or cached.get("verification") != "verified_root_cache"
            or digest(body) != cached.get("content_digest")
            or content_inventory(directory, exclude=("root_cache_manifest.json",)) != cached.get("files")):
        raise ValueError("Root cache content digest changed; refusing reuse")
    if expected_digest is not None and expected_digest != cached["content_digest"]:
        raise ValueError("Root cache differs from the run's frozen digest")
    if file_hash(directory / "evidence/run_manifest.json") != cached["baseline_manifest_sha256"]:
        raise ValueError("Root baseline manifest digest mismatch")
    baseline = json.loads((directory / "evidence/run_manifest.json").read_text())
    if (baseline["reproducibility_fingerprint"] != cached["baseline_fingerprint"] or
            digest(fingerprint_inputs(baseline)) != cached["baseline_fingerprint"] or
            compatibility(baseline) != cached["compatibility"]):
        raise ValueError("Root baseline evidence identity mismatch")
    seal = json.loads((directory / "evidence/baseline_seal.json").read_text())
    if digest(seal["files"]) != seal["content_digest"] or seal["content_digest"] != cached["baseline_content_digest"]:
        raise ValueError("Root baseline content seal mismatch")
    for name, facts in cached["files"].items():
        original = (name.replace("initial/predictions/baseline/", "predictions/", 1)
                    if name.startswith("initial/predictions/baseline/") else
                    name.removeprefix("initial/") if name.startswith("initial/src/") else
                    name.removeprefix("evidence/") if name.startswith("evidence/") else None)
        if original is not None and original != "baseline_seal.json" and seal["files"].get(original) != facts:
            raise ValueError("Imported file differs from the baseline content seal")
    shared = cached["compatibility"]
    if source_snapshot(directory / "initial/src") != shared["source"]:
        raise ValueError("Imported initial source hash mismatch")
    results = [validate_task_result(directory / "initial/predictions/baseline", t, directory / "evidence/codex_calls",
                                   policy_hash=shared["policy_sha256"], model=shared["model"], reasoning=shared["reasoning_effort"])
               for t in shared["ordered_task_ids"]]
    if results != cached["per_task_results"]:
        raise ValueError("Imported per-task result digest mismatch")
    performance = json.loads((directory / "initial/metadata.json").read_text())["overall_performance"]
    resolved = [r["task_id"] for r in results if r["result"] == "resolved"]
    unresolved = [r["task_id"] for r in results if r["result"] == "unresolved"]
    count = len(shared["ordered_task_ids"])
    if (performance["total_submitted_ids"] != shared["ordered_task_ids"] or performance["total_submitted_instances"] != count
            or performance["total_resolved_ids"] != resolved or performance["total_unresolved_ids"] != unresolved
            or performance["total_resolved_instances"] != len(resolved) or performance["accuracy_score"] != len(resolved)/count
            or cached["resolved"] != len(resolved) or cached["unresolved"] != len(unresolved)):
        raise ValueError("Imported root counts disagree with per-task results")
    if target is not None:
        wanted = compatibility(target)
        for key in wanted:
            if key == "creator_commit":
                # Same rule as validate_baseline: an ancestor commit, all other fields exact.
                if not _commit_is_ancestor(cached["compatibility"].get(key), wanted[key]):
                    raise ValueError("Root compatibility mismatch: " + key)
            elif cached["compatibility"].get(key) != wanted[key]:
                raise ValueError("Root compatibility mismatch: " + key)
    return cached


def root_reference(cached):
    return {"verification": "verified_root_cache", "root_cache_digest": cached["content_digest"],
            "baseline_manifest_sha256": cached["baseline_manifest_sha256"],
            "baseline_fingerprint": cached["baseline_fingerprint"],
            "resolved": cached["resolved"], "completed": cached["resolved"] + cached["unresolved"],
            "accuracy": cached["resolved"] / (cached["resolved"] + cached["unresolved"])}


def prepare_search_root(config, output, proposed):
    """Validate before any search operation, and return the immutable root input."""
    output = Path(output)
    if config.paths.initial_baseline_dir:
        destination = config.paths.root_cache_dir or str(output / ".root-cache")
        import_baseline(config.paths.initial_baseline_dir, destination, proposed)
        config.paths.root_cache_dir = destination
    if config.paths.root_cache_dir:
        directory = Path(config.paths.root_cache_dir)
        cached = verify_root_cache(directory, proposed)
        initial, reference = directory / "initial", root_reference(cached)
    elif config.protocol.effective_name == "smoke" and config.protocol.allow_legacy_initial_cache:
        initial = Path("initial_polyglot" if config.evaluation.polyglot else "initial_swe") / config.paths.initial_agent_name
        if not (initial / "metadata.json").is_file():
            raise ValueError("Explicit legacy smoke cache is missing; root evaluation is disabled")
        performance = json.loads((initial / "metadata.json").read_text())["overall_performance"]
        if set(performance["total_submitted_ids"]) != set(proposed["tasks"]["ordered_ids"]):
            raise ValueError("Legacy smoke cache task membership mismatch")
        reference = {"verification": "legacy_unverified_cache", "root_cache_digest": digest(content_inventory(initial)),
                     "baseline_manifest_sha256": None, "baseline_fingerprint": None,
                     "resolved": performance["total_resolved_instances"], "completed": performance["total_submitted_instances"],
                     "accuracy": performance["accuracy_score"]}
    else:
        raise ValueError("Corrected HGM/WGM requires --root-cache-dir or --initial-baseline-dir; legacy caches require explicit smoke opt-in")
    copied = output / "initial"
    if copied.exists() and content_inventory(copied) != content_inventory(initial):
        raise ValueError("Imported root copy changed; refusing resume")
    return initial, reference


def verify_imported_root(initial, copied, reference=None):
    """Recheck the sealed input after copying, closing the ordinary copy race."""
    initial, copied = Path(initial), Path(copied)
    if (initial.parent / "root_cache_manifest.json").is_file():
        cached = verify_root_cache(initial.parent, expected_digest=(reference or {}).get("root_cache_digest"))
        expected = {name.removeprefix("initial/"): facts for name,facts in cached["files"].items() if name.startswith("initial/")}
    else:
        expected = content_inventory(initial)
        if reference and digest(expected) != reference["root_cache_digest"]:
            raise ValueError("Legacy smoke cache changed while copying")
    if content_inventory(copied) != expected:
        raise ValueError("Root input changed during copy")
