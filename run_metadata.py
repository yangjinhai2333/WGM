"""Immutable experiment identity and atomic, reconstructible run accounting.

No model calls, image builds, login files or complete environment snapshots.
Call records are individually replaced by ID; resume never adds old totals again.
"""

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import warnings
import pathspec

from task_schedule import pool_size
from candidate_workspace import _atomic_write, _forbidden, _git
from config import effective_worker_count
from runners.codex_exec import _clean_environment
from runners.usage import aggregate_usage
from network_readiness import safe_command as safe_readiness_command

ROOT = Path(__file__).resolve().parent
CATEGORIES = ("resolved", "unresolved", "empty_patch", "error", "incomplete")
PACKAGES = ("swebench", "numpy", "PyYAML", "datasets", "docker", "pathspec")
PROMPTS = ("prompts/self_improvement_prompt.py", "prompts/testrepo_prompt.py",
           "candidate_workspace.py", "self_improve_step.py", "downstream_solver.py")


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value, *, exclusive=False):
    path = Path(path)
    data = encoded(value) + b"\n"
    if not exclusive:
        _atomic_write(path, data)
        return
    # Linking a fully flushed temporary file is atomic AND never overwrites.
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".manifest-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def repository_snapshot(root):
    root = Path(root)
    dirty = bool(_git(root, "status", "--porcelain", "--untracked-files=no").strip())
    diff = _git(root, "diff", "HEAD", "--binary", "--no-ext-diff", "--no-textconv") if dirty else b""
    return {
        "commit": _git(root, "rev-parse", "HEAD").decode().strip(),
        "branch": _git(root, "branch", "--show-current").decode().strip() or None,
        "tracked_clean": not dirty,
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest() if dirty else None,
        "untracked_paths": sorted(p.decode() for p in _git(root, "ls-files", "--others", "--exclude-standard", "-z").split(b"\0") if p),
    }


def check_repository(repository, *, require_clean=True):
    if require_clean and not repository["tracked_clean"]:
        raise ValueError("Tracked worktree is dirty; review and commit changes before running a baseline.")
    paths = repository["untracked_paths"]
    if paths:
        warnings.warn("Untracked paths recorded (contents not read): " + ", ".join(paths), stacklevel=2)
    active = [p for p in paths if p != "test.py" and not _forbidden(p)
              and (p.endswith((".py", ".yaml", ".yml", ".toml")) or p.startswith("prompts/")
                   or p in (".dockerignore", "requirements.txt", "pytest.ini")
                   or "/subsets/" in p)]
    if active:
        raise ValueError("Untracked runtime inputs must be committed: " + ", ".join(active))


def source_snapshot(source, *, root=ROOT):
    """Hash source files only; never read any main-repository untracked content."""
    files = source_inventory(source, root=root)
    return {"sha256": digest(files), "file_count": len(files)}


def source_inventory(source, *, root=ROOT):
    source, root = Path(source).resolve(), Path(root).resolve()
    if not source.is_dir():
        raise ValueError("Initial agent source is missing")
    if source == root:
        names = [p.decode() for p in _git(root, "ls-files", "-z").split(b"\0") if p]
        ignore_file = root / ".dockerignore"
        ignore = pathspec.PathSpec.from_lines("gitwildmatch", ignore_file.read_text().splitlines() if ignore_file.is_file() else [])
        names = [name for name in names if not ignore.match_file(name)]
    else:
        names = [p.relative_to(source).as_posix() for p in source.rglob("*") if p.is_file()]
    files = {}
    for name in sorted(names):
        path = source / name
        if name == "test.py" or _forbidden(name) or path.is_symlink() or not path.is_file():
            continue
        files[name] = {"sha256": file_hash(path), "executable": bool(path.stat().st_mode & 0o111)}
    return files


def copy_source_snapshot(source, destination, expected, *, root=ROOT):
    """Copy exactly the manifest's source inputs, without executing that source."""
    source, destination = Path(source), Path(destination)
    files = source_inventory(source, root=root)
    if {"sha256": digest(files), "file_count": len(files)} != expected:
        raise ValueError("Initial source snapshot changed")
    destination.mkdir(parents=True, exist_ok=False)
    for name, facts in files.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((source / name).read_bytes())
        target.chmod(0o755 if facts["executable"] else 0o644)
    if source_snapshot(destination, root=root) != expected:
        raise ValueError("Copied initial source snapshot differs")


def evaluator_identity(root=ROOT):
    root = Path(root)
    names = ("downstream_solver.py", "swe_bench/harness.py", "swe_bench/report.py",
             "polyglot/harness.py")
    value = {"files": {name: file_hash(root / name) if (root / name).is_file() else None for name in names}}
    value["packages"] = {}
    for name in ("swebench", "datasets", "docker"):
        try:
            value["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            value["packages"][name] = None
    return {**value, "sha256": digest(value)}


def version(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10, env=_clean_environment())
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def environment_snapshot(config, output):
    binaries = {name: shutil.which(config.codex.executable if name == "codex" else name)
                for name in ("node", "codex", "git", "docker")}
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    memory = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                memory = int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    disk_path = Path(output).resolve()
    while not disk_path.exists():
        disk_path = disk_path.parent
    docker_version = version(["docker", "version", "--format", "{{json .}}"])
    try:
        docker_version = json.loads(docker_version or "null") or {}
    except ValueError:
        docker_version = {}
    proxies = {name: name in os.environ for name in
               ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "all_proxy", "no_proxy")}
    provider_names = {"OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OpenRouter_API_KEY", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
                      "AWS_SESSION_TOKEN", "AWS_BEARER_TOKEN_BEDROCK", "ANTHROPIC_AUTH_TOKEN", "OPENAI_AUTH_TOKEN",
                      "OPENROUTER_API_KEY", "AZURE_OPENAI_API_KEY", "GOOGLE_APPLICATION_CREDENTIALS"}
    provider_names.update(k for k in os.environ if "APIKEY" in k.upper().replace("_", ""))
    return {
        "python": {"executable": sys.executable, "version": platform.python_version()},
        "node": {"executable": binaries["node"], "version": version([binaries["node"], "--version"]) if binaries["node"] else None},
        "codex": {"executable": binaries["codex"], "version": version([binaries["codex"], "--version"]) if binaries["codex"] else None},
        "git_version": version(["git", "--version"]),
        "docker_client_version": docker_version.get("Client", {}).get("Version"),
        "docker_server_version": docker_version.get("Server", {}).get("Version"),
        "packages": packages, "os": platform.system(), "kernel": platform.release(),
        "architecture": platform.machine(), "cpu_count": os.cpu_count(),
        "available_memory_bytes": memory, "available_disk_bytes": shutil.disk_usage(disk_path).free,
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"), "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "proxy_present": proxies, "provider_credentials_present": {k: k in os.environ for k in sorted(provider_names)},
    }


def effective_configuration(config, run_type=None, effective_branch_count=None):
    value = config.to_dict()
    value.pop("network_readiness", None)  # Runtime configuration never enters scientific identity.
    if run_type is not None:
        from search_config import experiment
        identity = experiment(config, run_type, effective_branch_count)
        value.pop("wgm", None)
        value.pop("wgm_v2", None)
        value.pop("dgm", None)
        value.pop("hgm_incremental", None)  # Target is runtime control; experiment freezes the horizon.
        value.pop("search", None)
        if identity["algorithm"] != "baseline":
            value["search"] = identity
            if identity["algorithm"] in ("wgm_v2", "dgm"):
                value.pop("optimization", None)
            else:
                value["optimization"] = identity["optimization"]
        corrected = identity.get("hgm_profile") == "paper_corrected" or identity.get("shared_hgm_profile") == "paper_corrected"
        if identity["algorithm"] == "wgm_v2":
            value["protocol"] = identity["protocol"]
            value["execution"]["max_task_evals"] = config.wgm_v2.planning_horizon
            value["execution"]["max_workers"] = 1
        elif identity["algorithm"] == "dgm":
            value["execution"]["max_task_evals"] = 800
            value.pop("protocol", None)
            value.pop("diagnosis_context", None)
        elif corrected:
            value["protocol"] = identity["protocol"]
            value["execution"]["max_task_evals"] = identity["search_budget"]
        else:
            value.pop("protocol", None)
            value.pop("diagnosis_context", None)
            for key in ("root_cache_dir", "initial_baseline_dir"):
                value["paths"].pop(key, None)
    value["effective"] = {
        "workers": effective_worker_count(config.execution.max_workers, config.llm.self_improve_backend,
                                          config.llm.downstream_backend, config.llm.diagnose_backend),
        "timeouts": {"diagnosis": config.codex.timeout_seconds,
                     "self_improvement": config.codex.timeout_seconds,
                     "downstream": config.execution.evaluation_timeout},
    }
    return value


def provider_prompt_identity(provider):
    """Digest of provider-specific prompt text; absent for the original Codex path."""
    from provider_prompts import identity
    return identity(provider)


def fingerprint_inputs(manifest):
    from execution_policy import semantics
    configuration = deepcopy(manifest["configuration"]["effective"])
    configuration.pop("network_readiness", None)
    configuration.get("execution", {}).pop("evaluation_workers", None)
    # Output destinations and resume locations do not change the experiment.
    for key in ("output_dir", "continue_from", "root_cache_dir", "initial_baseline_dir"):
        configuration.get("paths", {}).pop(key, None)
    for key in ("dataset_path", "polyglot_dataset_path"):
        configuration.get("evaluation", {}).pop(key, None)
    configuration.get("codex", {}).pop("executable", None)
    return {
        "run_type": manifest["run_type"],
        **({"execution_policy": semantics(manifest["execution_policy"])} if "execution_policy" in manifest else {}),
        **({"evaluation_execution": manifest["evaluation_execution"]} if "evaluation_execution" in manifest else {}),
        **({"experiment": manifest["experiment"]} if "experiment" in manifest else {}),
        **({"v2_sampling": manifest["v2_sampling"]} if "v2_sampling" in manifest else {}),
        **({"dgm_inputs": manifest["dgm_inputs"]} if "dgm_inputs" in manifest else {}),
        **({"hgm_incremental_inputs": manifest["hgm_incremental_inputs"]}
           if "hgm_incremental_inputs" in manifest else {}),
        **({"v2_derived": manifest["v2_derived"]} if "v2_derived" in manifest else {}),
        **({"comparison_final": manifest["comparison_final"]} if "comparison_final" in manifest else {}),
        "commit": manifest["repository"]["commit"],
        "tracked_diff_sha256": manifest["repository"]["tracked_diff_sha256"],
        "configuration": configuration, "configuration_sha256": digest(configuration),
        "tasks": manifest["tasks"], "policy": manifest["policy"], "dataset": manifest["dataset"],
        **({"evaluator": manifest["evaluator"]} if "evaluator" in manifest else {}),
        **({"root_initialization": manifest["root_initialization"]} if "root_initialization" in manifest else {}),
        "configured_model": manifest["codex"]["configured_model"],
        "configured_reasoning_effort": manifest["codex"]["configured_reasoning_effort"],
        "codex_cli_version": manifest["codex"]["codex_cli_version"],
        **({"provider_prompt_identity": manifest["codex"]["prompt_identity"]}
           if "prompt_identity" in manifest["codex"] else {}),
        "runtime_versions": {k: manifest["environment"].get(k) for k in ("packages", "git_version", "docker_server_version", "architecture")},
        "python_version": manifest["environment"]["python"]["version"],
        "node_version": manifest["environment"]["node"]["version"],
    }


def origin_network_semantics(manifest, readiness):
    """Runtime qualification only; never relax a scientific/resume identity."""
    if (manifest or {}).get("execution_policy", {}).get("effective_policy") != "origin_hgm_v1":
        return {}
    guarded = bool(readiness.get("ever_enabled", readiness.get("enabled", False)))
    return {"origin_network_guarded": guarded,
            "origin_equivalence": "completed_runs_only" if guarded else "original",
            **({"origin_partial_state_recovery": "not_supported"} if guarded else {})}


def build_manifest(config, output, run_type, task_ids, *, source=None, root=ROOT,
                   command=None, environment=None, dataset=None, require_clean=True,
                   require_pinned_model=False, patch_paths=(), shuffle=False, seed=None,
                   effective_branch_count=None, root_initialization=None):
    from task_schedule import unique_tasks, schedule_identity
    task_ids = unique_tasks(task_ids)
    root, source = Path(root), Path(source or root)
    if run_type == "wgm_v2":
        from wgm_v2_config import validate_initial_source
        validate_initial_source(config, source)
    repository = repository_snapshot(root)
    check_repository(repository, require_clean=require_clean)
    if require_pinned_model and not config.codex.model:
        raise ValueError("--require-pinned-model requires an explicit codex.model / --codex-model")
    if not config.codex.model:
        warnings.warn("MODEL NOT PINNED: account default is unresolved; configure codex.model for reproducibility", stacklevel=2)
    env = environment if environment is not None else environment_snapshot(config, output)
    folder = root / ("polyglot" if config.evaluation.polyglot else "swe_bench")
    marker_path = root / config.evaluation.dataset_path / "wgm_dataset.json"
    if dataset is None:
        if config.evaluation.polyglot:
            data_path = root / config.evaluation.polyglot_dataset_path
            dataset = {"name": "polyglot", "split": None, "entries": len(json.loads(data_path.read_text())),
                       "revision": None, "fingerprint": file_hash(data_path)}
        else:
            marker = json.loads(marker_path.read_text())
            from datasets import load_from_disk
            prepared = load_from_disk(str(root / config.evaluation.dataset_path))
            dataset = {"name": config.evaluation.dataset_name, "split": marker.get("split"),
                       "entries": len(prepared), "revision": marker.get("revision"),
                       "fingerprint": getattr(prepared, "_fingerprint", None)}
    dataset = dict(dataset)
    try:
        dataset["swebench_checkout_commit"] = _git(root / "swe_bench/SWE-bench", "rev-parse", "HEAD").decode().strip()
    except Exception:
        dataset["swebench_checkout_commit"] = None
    configuration = effective_configuration(config, run_type, effective_branch_count)
    configuration["paths"]["output_dir"] = str(Path(output).resolve())
    # Validation argv is an explicit local command, never an environment dump.
    policy = {"downstream_instructions_sha256": file_hash(source / "prompts/downstream_instructions.txt"),
              "prompt_files_sha256": {name: file_hash(root / name) for name in PROMPTS if (root / name).is_file()},
              "initial_source_snapshot": source_snapshot(source, root=root),
              "ancestor_patch_sha256": [file_hash(p) for p in patch_paths]}
    model = config.codex.model or None
    manifest = {
        "schema_version": 1, "run_id": Path(output).name, "run_type": run_type,
        "experiment": configuration.get("search", {"algorithm": "baseline"}),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "timezone": os.environ.get("TZ") or str(datetime.now().astimezone().tzinfo),
        "command": safe_readiness_command(command if command is not None else sys.argv), "repository": repository,
        "environment": env, "codex": {"configured_model": model,
            "configured_reasoning_effort": config.codex.reasoning_effort, "observed_model": None,
            "observed_reasoning_effort": None,
            "model_source": "project_config" if model else "account_default_unresolved",
            "codex_cli_version": env["codex"]["version"],
            # Codex manifests keep their historical keys; other providers are explicit.
            **({"provider": config.codex.provider, "prompt_identity": provider_prompt_identity(config.codex.provider)}
               if getattr(config.codex, "provider", None) else {})},
        "dataset": dataset,
        "evaluator": evaluator_identity(root),
        "tasks": {"ordered_ids": list(task_ids), "count": len(task_ids), "sha256": digest(list(task_ids)),
                  "subset_sha256": {name: file_hash(folder / "subsets" / (name + ".json"))
                                    for name in ("small", "medium") if (folder / "subsets" / (name + ".json")).is_file()},
                  "shuffle": shuffle, "random_seed": seed,
                  **schedule_identity(task_ids, seed if seed is not None else config.evaluation.task_order_seed),
                  "order_semantics": "adaptive RSI selection from this ordered pool" if run_type in ("hgm", "wgm") else "execution_order"},
        "configuration": {"effective": configuration, "sha256": digest(configuration)}, "policy": policy,
        "docker": {"base_images": None, "environment_images": None, "instance_images": None,
                   "source": "actual IDs recorded in task metadata after preparation; no extra builds"},
    }
    if root_initialization is not None:
        manifest["root_initialization"] = root_initialization
    elif manifest["experiment"].get("hgm_profile") == "legacy_code":
        manifest["root_initialization"] = {"verification": "legacy_unverified_cache"}
    from execution_policy import record, validated
    manifest["network_readiness"] = config.network_readiness.safe()
    manifest["execution_policy"] = record(config)
    from evaluation_batch import runtime_workers
    manifest["evaluation_parallelism"] = runtime_workers(config, run_type, len(task_ids))
    manifest.update(origin_network_semantics(manifest, manifest["network_readiness"]))
    if validated(config):
        from evaluation_plan import execution_identity
        manifest["evaluation_execution"] = execution_identity(config, manifest)
    manifest["fingerprint_inputs"] = fingerprint_inputs(manifest)
    manifest["reproducibility_fingerprint"] = digest(manifest["fingerprint_inputs"])
    return manifest


def finalize_search_manifest(recorder, identity):
    """Resolve initialization-dependent branch count; retain the original manifest machinery."""
    manifest = deepcopy(recorder.manifest)
    manifest["experiment"] = identity
    manifest["configuration"]["effective"]["search"] = identity
    manifest["configuration"]["sha256"] = digest(manifest["configuration"]["effective"])
    manifest["fingerprint_inputs"] = fingerprint_inputs(manifest)
    manifest["reproducibility_fingerprint"] = digest(manifest["fingerprint_inputs"])
    if "diagnosis_context_usage" in manifest:
        manifest["runtime_fingerprint"] = runtime_fingerprint(manifest)
    atomic_json(recorder.directory / "run_manifest.json", manifest)
    recorder.manifest = manifest


def differences(old, new, prefix=""):
    if isinstance(old, dict) and isinstance(new, dict):
        result = []
        for key in sorted(old.keys() | new.keys()):
            result.extend(differences(old.get(key), new.get(key), f"{prefix}.{key}" if prefix else key))
        return result
    return [prefix] if old != new else []


def open_manifest(directory, proposed, *, resume=False):
    path = Path(directory) / "run_manifest.json"
    if resume:
        if not path.is_file():
            raise ValueError("legacy/unverified result has no run_manifest.json; strict resume requires a new output directory")
        old = json.loads(path.read_text())
        if digest(old["fingerprint_inputs"]) != old["reproducibility_fingerprint"]:
            raise ValueError("Saved manifest fingerprint is inconsistent")
        if differences(fingerprint_inputs(old), old["fingerprint_inputs"]):
            raise ValueError("Saved manifest fields and fingerprint inputs disagree")
        if "runtime_fingerprint" in old:
            calls = [json.loads(p.read_text()) for p in (Path(directory) / "codex_calls").glob("*.json")]
            if runtime_fingerprint(old) != old["runtime_fingerprint"] or old["diagnosis_context_usage"] != context_usage(calls):
                raise ValueError("Saved diagnosis context execution fingerprint differs")
        changed = differences(old["fingerprint_inputs"], proposed["fingerprint_inputs"])
        if changed:
            raise ValueError("Resume fingerprint differs in: " + ", ".join(changed))
        return old
    atomic_json(path, proposed, exclusive=True)
    return proposed


def artifact_paths(directory, filename):
    """Visit pipeline artifacts, never files inside exported or candidate source."""
    paths = []
    for folder, children, files in os.walk(directory):
        children[:] = [name for name in children if name not in ("src", ".git", "agent", "evidence", ".root-cache")
                       and not name.startswith((".task-", ".candidate-"))]
        folder = Path(folder)
        is_task = folder.parent.name == "tasks"
        if filename in files and ((filename == "metadata.json" and is_task)
                                  or (filename == "batch.json" and "tasks" in children)):
            paths.append(folder / filename)
        if is_task:
            children.clear()  # Only metadata at this level belongs to the pipeline.
    return sorted(paths)


def task_records(directory):
    records = []
    for path in artifact_paths(directory, "metadata.json"):
        metadata = json.loads(path.read_text())
        prediction_path = path.parent.parent.parent / (path.parent.name + ".json")
        prediction = json.loads(prediction_path.read_text()) if prediction_path.is_file() else {}
        kind = prediction.get("eval_result", "incomplete")
        if metadata.get("interrupted") and kind == "error":
            kind = "incomplete"
        if kind not in CATEGORIES:
            raise ValueError("Unknown task classification in " + str(path))
        if prediction and bool(prediction.get("success")) != (kind in ("resolved", "unresolved")):
            raise ValueError("Task success/classification disagree in " + str(path))
        records.append({"category": kind, "metadata": metadata, "key": str(path.relative_to(directory))})
    return records


def context_usage(calls):
    return [{"call_id": r["call_id"], **r["diagnosis_context"]} for r in sorted(calls, key=lambda r:r["call_id"])
            if "diagnosis_context" in r]


def runtime_fingerprint(manifest):
    # Observed budgets belong to a separate execution fingerprint: changing the
    # resume identity during an in-flight operation would invalidate checkpoints.
    return digest({"reproducibility_fingerprint": manifest["reproducibility_fingerprint"],
                   "diagnosis_context_usage": manifest["diagnosis_context_usage"]})


def protocol_report(manifest, state):
    """Read existing experiment/checkpoint evidence; never alter search state."""
    experiment = (manifest or {}).get("experiment", {})
    protocol = ("baseline" if experiment.get("algorithm") == "baseline" else
                experiment.get("protocol", {}).get("name", "legacy/unavailable"))
    if experiment.get("algorithm") == "wgm_v2":
        protocol = manifest.get("wgm_v2_runtime", {}).get("protocol", "unavailable")
    budget = experiment.get("search_budget")
    metrics, warnings = None, []
    if experiment.get("algorithm") == "wgm" and state is not None:
        selector = state.get("selector") or {}
        records = selector.get("feedback", {})
        selections = [{k: row.get(k) for k in ("expansion_id", "selected_branch_id", "parent_id",
                       "candidate_id", "selection_probability")}
                      for row in state.get("history", [])
                      if row.get("phase") == "search" and row.get("selection_probability") is not None]
        updates = [{k: r.get(k) for k in ("expansion_id", "selected_branch_id", "candidate_id",
                    "selection_probability", "reward", "importance_weighted_reward", "config")}
                   for r in records.values() if r["updated"]]
        metrics = {"branch_selection_count": len(selections), "branch_selections": selections,
                   "weight_update_count": len(updates), "weight_updates": updates,
                   "log_weights": deepcopy(selector.get("log_weights", {})),
                   "completed_feedback": len(updates),
                   "pending_feedback": sum(not r["updated"] for r in records.values()),
                   **{k: state.get("summary", {}).get(k) for k in
                      ("reproductive_lineage_L", "design_diversity_D", "design_diversity_reason")}}
        if protocol == "pilot" and not selections:
            warnings.append("pilot budget did not exercise the WGM branch selector")
    return {"protocol": protocol, "search_budget": budget, "wgm_selector": metrics,
            "warnings": warnings}


def make_summary(directory, *, manifest=None, status=None, whole_run_total=None):
    from evaluation_batch import batch_reports
    directory = Path(directory)
    tasks = task_records(directory)
    counts = {kind + "_instances": sum(r["category"] == kind for r in tasks) for kind in CATEGORIES}
    submitted = len(tasks)
    completed = counts["resolved_instances"] + counts["unresolved_instances"]
    def rate(value):
        return value / submitted if submitted else None
    if sum(counts.values()) != submitted:
        raise ValueError("Task categories do not partition submitted instances")
    call_files = sorted((directory / "codex_calls").glob("*.json"))
    calls = [json.loads(p.read_text()) for p in call_files]
    legacy = manifest is None
    codex_usage = aggregate_usage(calls)
    if legacy:
        # Old results have a downstream counter but no reliable launch/usage data.
        for phase in codex_usage.values():
            for key in ("attempted_calls", "started_calls", "successful_calls", "failed_calls", "timed_out_calls", "duration_seconds"):
                phase[key] = None
        known = [r["metadata"].get("codex_calls") for r in tasks]
        if known and all(type(k) is int for k in known):
            codex_usage["downstream"]["attempted_calls"] = sum(known)
    times = [r["metadata"].get("timing", {}) for r in tasks]
    durations = sorted(t["per_task_total"] for t in times if isinstance(t.get("per_task_total"), (int, float)))
    if any(v < 0 or not math.isfinite(v) for t in times for v in t.values() if isinstance(v, (int, float))):
        raise ValueError("Invalid task timing")
    mapping = {"codex_execution": "codex_execution", "workspace_extraction": "task_workspace_extraction",
               "patch_handling": "patch_generation_application", "evaluation": "container_evaluation", "report": "report_generation"}
    timing = {}
    for name, key in mapping.items():
        values = [t.get(key) for t in times]
        timing[name] = sum(values) if values and all(isinstance(v, (int, float)) for v in values) else None
    if calls:
        # RSI also spends time diagnosing and generating candidates, which are
        # outside task metadata. The call ledger includes all three stages.
        timing["codex_execution"] = codex_usage["total"]["duration_seconds"]
    timing.update(whole_run_total=whole_run_total,
                  cold_image_preparation=sum(r["metadata"].get("timing", {}).get("dataset_image_preparation", 0)
                                             for r in tasks if r["metadata"].get("instance_image_cached") is False),
                  per_task_mean=statistics.mean(durations) if durations else None,
                  per_task_median=statistics.median(durations) if durations else None,
                  per_task_p90=durations[math.ceil(.9 * len(durations)) - 1] if durations else None,
                  fastest_task=durations[0] if durations else None, slowest_task=durations[-1] if durations else None,
                  cached_tasks=sum(r["metadata"].get("instance_image_cached") is True for r in tasks),
                  uncached_tasks=sum(r["metadata"].get("instance_image_cached") is False for r in tasks),
                  cache_status_unavailable=sum(r["metadata"].get("instance_image_cached") is None for r in tasks))
    measured = [r for r in tasks if isinstance(r["metadata"].get("timing", {}).get("per_task_total"), (int, float))]
    for name, choose in (("fastest_task", min), ("slowest_task", max)):
        task = choose(measured, key=lambda r: r["metadata"]["timing"]["per_task_total"]) if measured else None
        timing[name] = {"instance_id": task["metadata"]["instance_id"],
                        "duration_seconds": task["metadata"]["timing"]["per_task_total"]} if task else None
    batch_report_times = [json.loads(p.read_text()).get("timing", {}).get("report_generation")
                          for p in artifact_paths(directory, "batch.json")]
    if timing["report"] is not None:
        timing["report"] += sum(v for v in batch_report_times if isinstance(v, (int, float)))
    failures = {}
    for row in tasks:
        if (row["category"] not in ("resolved", "unresolved")
                and row["metadata"].get("status") != "running"):
            stage = row["metadata"].get("stage", "unavailable")
            failures[stage] = failures.get(stage, 0) + 1
    observed = sorted({r["observed_model"] for r in calls if r.get("observed_model")})
    efforts = sorted({r["observed_reasoning_effort"] for r in calls if r.get("observed_reasoning_effort")})
    codex = deepcopy(manifest["codex"]) if manifest else {key: None for key in
        ("configured_model", "configured_reasoning_effort", "observed_model", "observed_reasoning_effort", "codex_cli_version")}
    codex.update(observed_model=observed[0] if len(observed) == 1 else None, observed_models=observed,
                 observed_reasoning_effort=efforts[0] if len(efforts) == 1 else None)
    codex["model_source"] = "cli_event" if observed else "project_config" if codex.get("configured_model") else "account_default_unresolved"
    search_path = directory / "search_state.json"
    search_state = json.loads(search_path.read_text())["state"] if search_path.is_file() else None
    search_summary = search_state["summary"] if search_state is not None else None
    protocol_metrics = protocol_report(manifest, search_state)
    final_path = directory / "best_belief.json"
    final_eval_path = directory / "final_evaluation/summary.json"
    final_eval = json.loads(final_eval_path.read_text()) if final_eval_path.is_file() else None
    stages = {}
    for name in ("root", "search", "final_evaluation"):
        selected = [r for r in tasks if ("final_evaluation" if r["key"].startswith("final_evaluation/") else
                    "root" if r["key"].startswith("initial/") or (manifest and manifest.get("run_type") not in ("hgm", "wgm", "wgm_v2"))
                    else "search") == name]
        stages[name] = {"submitted": len(selected),
                        "completed": sum(r["category"] in ("resolved", "unresolved") for r in selected),
                        "resolved": sum(r["category"] == "resolved" for r in selected)}
    task_identity = manifest.get("tasks", {}) if manifest else {}
    schedule = {k: task_identity.get(k) for k in ("task_membership_hash", "task_order_hash", "task_order_seed", "schedule_schema")}
    schedule["verification"] = "recorded" if schedule["schedule_schema"] else "legacy/unavailable"
    root_initialization = manifest.get("root_initialization") if manifest else None
    if root_initialization is None and manifest and manifest.get("run_type") == "hgm":
        root_initialization = {"verification": "legacy_unverified_cache"}
    readiness_path = directory / "network_readiness.json"
    readiness = (json.loads(readiness_path.read_text()) if readiness_path.is_file()
                 else (manifest or {}).get("network_readiness", {}))
    return {"schema_version": 1, "status": status or "running", "verification": "legacy/unverified" if legacy else "manifest_verified",
            "evaluation_parallelism": (manifest or {}).get("evaluation_parallelism"),
            "evaluation_batches": batch_reports(directory),
            **({"network_readiness": readiness} if readiness else {}),
            **origin_network_semantics(manifest, readiness),
            "execution_policy": (manifest or {}).get("execution_policy", {}).get("effective_policy", "legacy/unavailable"),
            "task_recovery": (manifest or {}).get("execution_policy", {}).get("task_recovery", "legacy"),
            "statistical_guarantee": (manifest or {}).get("execution_policy", {}).get("statistical_guarantee", "not_validated_iid"),
            "experiment": manifest.get("experiment", {"algorithm": "hgm", "hgm_profile": "legacy_code"}
                                       if manifest.get("run_type") == "hgm" else {"algorithm": "baseline"}) if manifest else None,
            "search": search_summary, **protocol_metrics,
            "expansion_count": search_summary.get("expansion_count") if search_summary else None,
            "pilot_effect_comparison_eligible": (bool(final_eval and final_eval.get("final_evaluation_complete")
                and not protocol_metrics["warnings"] and (manifest.get("experiment", {}).get("algorithm") != "wgm"
                or protocol_metrics["wgm_selector"] is not None)) if protocol_metrics["protocol"] == "pilot" else None),
            "metric_scope": "aggregate_evaluation_events", "evaluation_events": stages,
            "task_schedule": schedule,
            "root_initialization": root_initialization,
            "root_baseline_accuracy": (stages["root"]["resolved"]/stages["root"]["submitted"]
                if stages["root"]["submitted"] and stages["root"]["completed"] == stages["root"]["submitted"]
                == pool_size(bool(manifest["configuration"]["effective"].get("evaluation", {}).get("polyglot"))) else None),
            "final_evaluation": final_eval,
            "final_evaluation_complete": final_eval["final_evaluation_complete"] if final_eval else False,
            "final_accuracy": final_eval["final_accuracy"] if final_eval else None,
            "diagnosis_context_usage": context_usage(calls),
            "runtime_fingerprint": manifest.get("runtime_fingerprint") if manifest else None,
            "effective_configuration": deepcopy(manifest["configuration"]["effective"]) if manifest else None,
            "final_selection": json.loads(final_path.read_text()) if final_path.is_file() else None,
            "submitted_instances": submitted, "completed_instances": completed, **counts,
            "resolved_rate": rate(counts["resolved_instances"]), "completion_rate": rate(completed),
            "empty_patch_rate": rate(counts["empty_patch_instances"]), "error_rate": rate(counts["error_instances"]),
            "timing": timing, "codex_usage": codex_usage, "codex": codex, "failure_stage_counts": failures,
            "reproducibility_fingerprint": manifest["reproducibility_fingerprint"] if manifest else None,
            "docker": [{"instance_id": r["metadata"]["instance_id"],
                        "instance_image_cached": r["metadata"].get("instance_image_cached"),
                        "images": r["metadata"]["docker"]} for r in tasks if r["metadata"].get("docker")],
            "executed_task_order": ([r["metadata"].get("instance_id") for r in sorted(tasks, key=lambda r: r["metadata"]["submitted_at_ns"])]
                                    if all("submitted_at_ns" in r["metadata"] for r in tasks) else None)}


class RunRecorder:
    """Per-run ownership, no module-global counters; each update replaces state."""
    def __init__(self, directory, manifest, *, initial_elapsed=0.0, network_readiness=None):
        self.directory, self.manifest = Path(directory), manifest
        self.lock = threading.RLock()
        self.started = time.monotonic()
        previous = self.directory / "run_summary.json"
        self.previous_total = json.loads(previous.read_text())["timing"].get("whole_run_total", 0) if previous.is_file() else 0
        self.previous_total = (self.previous_total or 0) + initial_elapsed
        self.status = "running"
        if network_readiness is not None and (network_readiness.enabled or
                                             (self.directory / "network_readiness.json").exists()):
            # Resuming may change runtime configuration without changing identity.
            # Preserve cumulative counters and the fact that this run was guarded.
            from network_readiness import COUNTERS
            self.record_readiness({"schema_version": 1, "enabled": network_readiness.enabled,
                "configuration": network_readiness.safe(), **dict.fromkeys(COUNTERS, 0)})
        # A crash may leave a start record; retain it as a failed/interrupted call.
        for path in (self.directory / "codex_calls").glob("*.json"):
            record = json.loads(path.read_text())
            if record["status"] in ("attempting", "running"):
                record.update(status="interrupted", failed_calls=1)
                atomic_json(path, record)
        self.refresh()

    def record_call(self, record):
        with self.lock:
            folder = self.directory / "codex_calls"
            folder.mkdir(exist_ok=True)
            atomic_json(folder / (record["call_id"] + ".json"), record)
            if "diagnosis_context" in record:
                self.manifest["diagnosis_context_usage"] = context_usage(
                    [json.loads(p.read_text()) for p in folder.glob("*.json")])
                self.manifest["runtime_fingerprint"] = runtime_fingerprint(self.manifest)
                atomic_json(self.directory / "run_manifest.json", self.manifest)
            # DGM siblings share this lock. Rebuilding the report between
            # readiness and Popen can expire another worker's short receipt.
            # Records stay durable; report/controller exit reconstruct totals.
            if self.manifest.get('run_type') != 'dgm':
                self.refresh()

    def refresh(self, status=None):
        with self.lock:
            if status is not None:
                self.status = status
            summary = make_summary(self.directory, manifest=self.manifest, status=self.status,
                                   whole_run_total=(self.previous_total or 0) + time.monotonic() - self.started)
            atomic_json(self.directory / "run_summary.json", summary)
            return summary

    def record_readiness(self, row):
        from network_readiness import COUNTERS
        with self.lock:
            path = self.directory / "network_readiness.json"
            old = json.loads(path.read_text()) if path.exists() else dict.fromkeys(COUNTERS, 0)
            value = {**old, **row, **{key: old.get(key, 0) + row[key] for key in COUNTERS},
                     "ever_enabled": bool(old.get("ever_enabled", old.get("enabled", False)) or row["enabled"])}
            atomic_json(path, value)
            if row.get("timeout_count", 0) or row.get("cancel_count", 0):
                # Includes root initialization, before the outer search loop's
                # try/finally. A readiness stop is never a completed origin run.
                self.refresh("incomplete")


def read_report(directory):
    """Read-only, including old single-task directories with missing provenance."""
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError("Run directory does not exist")
    manifest_path = directory / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
    if manifest and manifest.get('experiment', {}).get('incremental') and 'comparison_final' not in manifest:
        from hgm_incremental import report
        return manifest, report(directory, manifest), {}
    if manifest and manifest.get('run_type') == 'dgm':
        from dgm_legacy_search import report
        return manifest, report(directory, manifest), {}
    if manifest and manifest.get("run_type") == "wgm_v2" and "v2_derived" not in manifest:
        from wgm_v2_report import read_search_report
        return manifest, read_search_report(directory, manifest), {}
    if manifest and manifest.get('run_type') in ('default-baseline', 'small-baseline', 'single-task') and (
            directory / 'evaluation_batches').is_dir():
        from evaluation_batch import read_fixed_report
        return manifest, read_fixed_report(directory, manifest), {}
    saved_path = directory / "run_summary.json"
    saved = json.loads(saved_path.read_text()) if saved_path.is_file() else None
    run_path = directory / "run.json"
    run = json.loads(run_path.read_text()) if run_path.is_file() else {}
    if manifest is None and not run:
        raise ValueError("Not a run directory: missing run_manifest.json and legacy run.json")
    status = saved.get("status") if saved else {"complete": "completed", "running": "running", "interrupted_or_failed": "failed"}.get(run.get("status"), "running")
    total = saved["timing"].get("whole_run_total") if saved else run.get("whole_run_total")
    summary = make_summary(directory, manifest=manifest, status=status, whole_run_total=total)
    if manifest and "runtime_fingerprint" in manifest:
        if (runtime_fingerprint(manifest) != manifest["runtime_fingerprint"] or
                manifest["diagnosis_context_usage"] != summary["diagnosis_context_usage"]):
            raise ValueError("Diagnosis context execution fingerprint disagrees with call records")
    if manifest is None:
        configured = run.get("config", {}).get("codex", {})
        summary["codex"]["configured_model"] = configured.get("model") or None
        summary["codex"]["configured_reasoning_effort"] = configured.get("reasoning_effort")
        if summary["codex"]["configured_model"]:
            summary["codex"]["model_source"] = "project_config"
    if status == "completed" and not summary["submitted_instances"]:
        raise ValueError("Completed run has no task metadata")
    if manifest and digest(manifest["fingerprint_inputs"]) != manifest["reproducibility_fingerprint"]:
        raise ValueError("Manifest fingerprint is inconsistent")
    if manifest and differences(fingerprint_inputs(manifest), manifest["fingerprint_inputs"]):
        raise ValueError("Manifest fields and fingerprint inputs disagree")
    if saved:
        keys = ("submitted_instances", "completed_instances", *(k + "_instances" for k in CATEGORIES))
        if any(saved.get(k) != summary[k] for k in keys):
            raise ValueError("run_summary.json counts disagree with task records")
    consistency = []
    for path in artifact_paths(directory, "batch.json"):
        batch = json.loads(path.read_text())
        local = task_records(path.parent)
        expected = {k + "_instances": sum(r["category"] == k for r in local) for k in CATEGORIES}
        expected.update(submitted_instances=len(local), completed_instances=sum(r["category"] in ("resolved", "unresolved") for r in local))
        mismatched = [k for k, v in expected.items() if batch.get("report", {}).get(k) != v]
        if mismatched:
            consistency.append(str(path.relative_to(directory)) + ": " + ", ".join(mismatched))
        # The existing baseline and node report writers place this same report
        # one or two levels above the prediction folder. Never rewrite it.
        for parent in (path.parent.parent, path.parent.parent.parent):
            report_path = parent / (path.parent.name + ".json")
            if not report_path.is_relative_to(directory) or not report_path.is_file():
                continue
            report = json.loads(report_path.read_text())
            mismatched = [k for k, v in expected.items() if report.get(k) != v]
            if mismatched:
                consistency.append(str(report_path.relative_to(directory)) + ": " + ", ".join(mismatched))
    if manifest and consistency and status in ("completed", "failed"):
        raise ValueError("Batch/task counts disagree: " + "; ".join(consistency))
    summary["consistency_warnings"] = consistency
    return manifest, summary, run


def report_text(manifest, summary, legacy_run=None):
    if summary.get('report_type') == 'hgm_incremental_search':
        from hgm_incremental import report_text as hgm_text
        return hgm_text(summary)
    if summary.get('report_type') == 'dgm_legacy_search':
        from dgm_legacy_search import report_text as dgm_text
        return dgm_text(summary)
    if summary.get('report_type') == 'fixed_evaluation_batch':
        from evaluation_batch import fixed_report_text
        return fixed_report_text(summary)
    if summary.get("report_type") == "wgm_v2_canonical_search":
        from wgm_v2_report import search_report_text
        return search_report_text(summary)
    manifest, legacy_run = manifest or {}, legacy_run or {}
    missing = lambda v: "unavailable" if v is None else str(v)
    c, t = summary["codex"], summary["timing"]
    lines = [f"Run type: {manifest.get('run_type', 'legacy/unverified')}", f"Status: {summary['status']}",
             f"Execution: execution_policy={summary.get('execution_policy', 'legacy/unavailable')}, "
             f"task_recovery={summary.get('task_recovery', 'legacy')}, "
             f"statistical_guarantee={summary.get('statistical_guarantee', 'not_validated_iid')}",
             f"Protocol: protocol={summary.get('protocol', 'legacy/unavailable')}, search_budget={missing(summary.get('search_budget'))}",
             f"Experiment: {json.dumps(summary.get('experiment'), sort_keys=True)}",
             f"WGM commit: {manifest.get('repository', {}).get('commit', 'unavailable')}",
             f"Reproducibility fingerprint: {missing(summary['reproducibility_fingerprint'])}",
             f"Configured/observed model: {missing(c['configured_model'])} / {missing(c['observed_model'])}",
             f"Reasoning effort (configured/observed): {missing(c['configured_reasoning_effort'])} / {missing(c['observed_reasoning_effort'])}",
             f"Codex CLI version: {missing(c['codex_cli_version'])}",
             f"Task set and hash: {manifest.get('tasks', {}).get('count', len(legacy_run.get('task_ids', [])))} / {manifest.get('tasks', {}).get('sha256', 'unavailable')}",
             f"Submitted/completed/resolved: {summary['submitted_instances']}/{summary['completed_instances']}/{summary['resolved_instances']}",
             f"Aggregate evaluation event resolved rate: {missing(summary['resolved_rate'])}",
             'Unresolved/empty/error/incomplete: ' + '/'.join(str(summary[k + '_instances']) for k in CATEGORIES[1:])]
    if "origin_network_guarded" in summary:
        lines.append("Origin network: origin_network_guarded=" + str(summary["origin_network_guarded"]).lower()
                     + ", origin_equivalence=" + summary["origin_equivalence"])
        if summary["origin_network_guarded"]:
            lines.append("Origin partial-state recovery: not_supported")
    if summary.get("network_readiness"):
        lines.append("Network readiness: " + json.dumps(summary["network_readiness"], sort_keys=True))
    for phase, usage in summary["codex_usage"].items():
        lines.append(f"Codex {phase}: attempted={missing(usage['attempted_calls'])}, started={missing(usage['started_calls'])}, successful={missing(usage['successful_calls'])}, failed={missing(usage['failed_calls'])}, timed_out={missing(usage['timed_out_calls'])}")
        lines.append(f"  Tokens input/cached/output: {missing(usage['input_tokens'])}/{missing(usage['cached_input_tokens'])}/{missing(usage['output_tokens'])}")
    if summary.get('evaluation_parallelism'):
        lines.append('Evaluation workers: ' + json.dumps(summary['evaluation_parallelism'], sort_keys=True))
    for batch in summary.get('evaluation_batches', []):
        lines.append('Fixed evaluation batch: ' + json.dumps(batch, sort_keys=True))
    lines.extend([f"Total/median/p90 seconds: {missing(t['whole_run_total'])}/{missing(t['per_task_median'])}/{missing(t['per_task_p90'])}",
                  f"Cold/cached/unknown images: {t['uncached_tasks']}/{t['cached_tasks']}/{t['cache_status_unavailable']}",
                  f"Failure-stage counts: {summary['failure_stage_counts']}"])
    if summary.get("search") is not None:
        lines.append("Search metrics: " + json.dumps(summary["search"], sort_keys=True))
    lines.append("Expansion count: " + missing(summary.get("expansion_count")))
    if summary.get("wgm_selector") is not None:
        lines.append("WGM EXP3: " + json.dumps(summary["wgm_selector"], sort_keys=True))
    for warning in summary.get("warnings", []):
        lines.append("WARNING: " + warning)
    if summary.get("protocol") == "pilot":
        lines.append("Pilot effect comparison eligible: " + str(summary["pilot_effect_comparison_eligible"]).lower())
    if summary.get("final_selection") is not None:
        lines.append("Best-belief: " + json.dumps(summary["final_selection"], sort_keys=True))
    for stage, label in (("root", "Root"), ("search", "Search event")):
        events = summary["evaluation_events"][stage]
        lines.append(f"{label} resolved/total: {events['resolved']}/{events['submitted']}")
    if summary.get("final_evaluation") is not None:
        final = summary["final_evaluation"]
        lines.append(f"Final evaluation resolved/requested: {missing(final.get('resolved_instances'))}/{missing(final.get('requested_tasks'))}")
    lines.extend(["Task schedule: " + json.dumps(summary["task_schedule"], sort_keys=True),
                  "Evaluation events by stage: " + json.dumps(summary["evaluation_events"], sort_keys=True),
                  "Root baseline accuracy: " + missing(summary["root_baseline_accuracy"]),
                  "Root initialization: " + json.dumps(summary.get("root_initialization"), sort_keys=True),
                  "Final evaluation complete: " + str(summary["final_evaluation_complete"]).lower(),
                  "Selected agent final full-set accuracy: " + missing(summary["final_accuracy"]),
                  "Final evaluation: " + json.dumps(summary.get("final_evaluation"), sort_keys=True),
                  "Diagnosis context budgets used: " + json.dumps(sorted(set(r["budget"] for r in summary["diagnosis_context_usage"]))),
                  "Runtime fingerprint: " + missing(summary.get("runtime_fingerprint"))])
    return "\n".join(lines)
