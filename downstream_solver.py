"""One host Codex call per benchmark task; Docker owns dependencies and grading.

The harness supplies benchmark-specific preparation/grading callbacks. This
module never imports an API client or executes an evolved legacy coding agent.
"""

from copy import copy
from dataclasses import dataclass, field, replace
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
import time
from network_readiness import ReadinessAbort, await_runner, task_completed

from candidate_workspace import (
    CandidateError, _CACHES, _atomic_write, _check_baseline, _copy_source, _forbidden, _git, _patch_paths,
)
from runners import CodexExecConfig, CodexExecRequest, CodexExecRunner
from runners.codex_exec import _stderr_summary
from runners.usage import invoke, aggregate_usage


TIMING_FIELDS = (
    "dataset_image_preparation", "task_workspace_extraction", "codex_execution",
    "patch_generation_application", "container_evaluation", "report_generation",
    "per_task_total",
)


@dataclass(frozen=True)
class TaskResult:
    success: bool
    prediction: dict = field(repr=False)
    metadata: dict = field(repr=False)


class TaskFailure(RuntimeError):
    """Fixed messages only: container output and task prompts are private."""


class ContainerCommandFailure(TaskFailure):
    """Retain argv and exit status for local sanitization, never raw output."""

    def __init__(self, argv, returncode):
        self.argv = list(argv)
        self.returncode = returncode
        super().__init__(f"Container command failed (exit status {returncode})")


def image_facts(client, spec):
    """Inspect existing images only, without building or exposing image env."""
    result = {}
    for level in ("base", "env", "instance"):
        name = getattr(spec, level + "_image_key", None)
        info = {"name": name if isinstance(name, str) else None, "id": None, "digests": None}
        if isinstance(name, str):
            try:
                image = client.images.get(name)
                info["id"] = image.id if isinstance(image.id, str) else None
                attrs = image.attrs if isinstance(image.attrs, dict) else {}
                info["digests"] = attrs.get("RepoDigests")
            except Exception:
                pass  # Unavailable provenance does not change benchmark results.
        result[level] = info
    return result


def put_bytes(container, destination, content):
    """Send a single owned artifact, without shell interpolation or credentials."""
    target = PurePosixPath(destination)
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        item = tarfile.TarInfo(target.name)
        item.size = len(content)
        # Benchmark images may evaluate as nonroot; scripts contain no secrets.
        item.mode = 0o644
        archive.addfile(item, io.BytesIO(content))
    if not container.put_archive(str(target.parent), stream.getvalue()):
        raise TaskFailure("Could not copy task artifact into container")


def container_command(container, argv, *, workdir="/testbed", check=True):
    # Match upstream SWE-bench's root patch application. Image users may be
    # nonroot while /testbed's Git metadata is owned by root.
    options = {"user": "root"} if argv[0] == "git" else {}
    result = container.exec_run(argv, workdir=workdir, **options)
    if check and result.exit_code != 0:
        raise ContainerCommandFailure(argv, result.exit_code)
    return result


def inspect_prepared_repository(container, base_commit):
    """Verify HEAD while preserving benchmark installation changes.

    SWE-bench may deliberately modify tracked files during pre_install (Sphinx
    setup.py dependency pins and tox.ini flags, for example). The host baseline
    must include that prepared state, just like the container used for grading.
    A nonempty diff is provenance, not an execution failure. Git errors and an
    incorrect HEAD still fail before the solver is called.
    """
    head = container_command(container, ["git", "rev-parse", "HEAD"]).output.decode().strip()
    expected = container_command(container, ["git", "rev-parse", base_commit]).output.decode().strip()
    if head != expected:
        raise TaskFailure("Prepared container is not at task base commit")
    diff = container_command(container, ["git", "diff", "--binary", "--no-ext-diff",
                                         "--no-textconv", expected, "--"]).output
    paths = container_command(container, ["git", "diff", "--name-only", "-z", "--no-ext-diff",
                                          "--no-textconv", expected, "--"]).output
    return {"base_commit": head, "tracked_changes": bool(diff),
            "tracked_diff_sha256": hashlib.sha256(diff).hexdigest() if diff else None,
            "tracked_changed_files": [p.decode("utf-8", errors="replace") for p in paths.split(b"\0") if p]}


def export_testbed(container, workspace, *, max_bytes=2 * 1024**3):
    """Extract /testbed safely; never restore its Git config, hooks or credentials.

    Stream the archive to disk; validate every member before writing it. Regular
    files and relative, in-tree symlinks are supported. No tar.extractall().
    """
    workspace.mkdir()
    stream, _ = container.get_archive("/testbed")
    with tempfile.TemporaryFile() as spool:
        received = 0
        for chunk in stream:
            received += len(chunk)
            if received > max_bytes:
                raise TaskFailure("Task archive exceeds workspace size limit")
            spool.write(chunk)
        spool.seek(0)
        links = []
        expanded = 0
        with tarfile.open(fileobj=spool, mode="r|*") as archive:
            for member in archive:
                path = PurePosixPath(member.name)
                if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "testbed":
                    raise TaskFailure("Unsafe task archive path")
                relative = PurePosixPath(*path.parts[1:])
                if str(relative) == ".":
                    continue
                if _forbidden(relative.as_posix()):
                    continue
                target = workspace / str(relative)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    expanded += member.size
                    if expanded > max_bytes:
                        raise TaskFailure("Extracted workspace exceeds size limit")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.extractfile(member) as source, target.open("xb") as output:
                        shutil.copyfileobj(source, output)
                    target.chmod(0o755 if member.mode & 0o111 else 0o644)
                elif member.issym():
                    # Defer links so no archive member can write through them.
                    links.append((target, member.linkname))
                else:
                    raise TaskFailure("Unsupported task archive entry")
        for target, link in links:
            if PurePosixPath(link).is_absolute() or not (target.parent / link).resolve().is_relative_to(workspace):
                raise TaskFailure("Unsafe task archive symlink")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(link)


# Build outputs agents create while running a Polyglot exercise locally
# (Gradle, Cargo, CMake, npm). They are never part of a solution.
_POLYGLOT_ARTIFACT_DIRS = {"build", ".gradle", "target", "node_modules", "CMakeFiles", "cmake-build-debug", ".kotlin"}


def remove_untracked_artifacts(workspace):
    """Polyglot only: delete untracked build outputs, binaries and forbidden paths.

    Untracked files only, plus tracked files on forbidden paths (exercise
    configuration such as .npmrc), which are restored to the baseline: neither
    is part of a solution. Every other tracked edit and new source file is kept.
    Returns the affected relative paths.
    """
    raw = _git(workspace, "ls-files", "-z", "--", ".")
    tracked_forbidden = [n.decode("utf-8", errors="replace") for n in raw.split(b"\0")
                         if n and _forbidden(n.decode("utf-8", errors="replace"))]
    if tracked_forbidden:
        _git(workspace, "checkout", "HEAD", "--", *tracked_forbidden)
    raw = _git(workspace, "ls-files", "--others", "-z", "--", ".")
    removed = list(tracked_forbidden)
    for name in (n.decode("utf-8", errors="replace") for n in raw.split(b"\0") if n):
        path = workspace / name
        if path.is_symlink():
            # e.g. node_modules -> /npm-install/node_modules copied from the test
            # commands; a solution never adds symlinks (the patch would be rejected).
            path.unlink()
            removed.append(name)
            continue
        if not path.is_file():
            continue
        parts = PurePosixPath(name).parts
        binary = b"\0" in path.read_bytes()[:8192]
        if binary or _forbidden(name) or any(part in _POLYGLOT_ARTIFACT_DIRS for part in parts[:-1]):
            path.unlink()
            removed.append(name)
    return sorted(removed)


def _capture_patch(workspace, baseline, max_patch_bytes):
    exclusions = [f":(glob,exclude)**/{name}/**" for name in sorted(_CACHES)]
    _git(workspace, "add", "--all", "--force", "--", ".", *exclusions)
    patch = _git(workspace, "diff", "--cached", "--no-ext-diff", "--no-textconv",
                 "--no-renames", "--src-prefix=a/", "--dst-prefix=b/", baseline,
                 "--", ".", *exclusions)
    if len(patch) > max_patch_bytes:
        raise TaskFailure("Task patch exceeds size limit")
    paths = _patch_paths(workspace, patch)
    for name in paths:
        if (workspace / name).is_symlink():
            raise TaskFailure("Task patch changes a symlink")
    return patch, paths


def task_prompt(policy, problem_statement, test_description):
    return (
        "Your current working directory is an isolated copy of the target repository.\n"
        "Modify production code to solve the task, not just tests. Do not create commits. "
        "Avoid unrelated changes. You may inspect files and run available lightweight commands. "
        "Do not invoke models, Docker, or install dependencies globally on the host. "
        "Missing dependencies are expected: formal evaluation runs in the prepared container. "
        "Do not access credentials or modify outputs or Git management files. "
        "The task patch is obtained from Git diff, including added and deleted files; "
        "do not return a textual patch.\n\n"
        f"<agent_instructions>\n{policy}\n</agent_instructions>\n\n"
        f"<problem_statement>\n{problem_statement}\n</problem_statement>\n\n"
        f"<test_description>\n{test_description}\n</test_description>\n"
        "Paths under /testbed in test instructions refer to your current directory.\n"
    )


def run_task(entry, out_dir, model_name, *, prepare, evaluate, cleanup,
             runner=None, codex_config=None, timeout=1800, policy="",
             max_patch_bytes=1024 * 1024, patch_filter=None, recorder=None,
             execution_spec=None, evaluation_ledger=None, lifecycle_callbacks=None, source_fingerprint=None):
    """Prepare, solve, apply and grade exactly once, cleaning only owned resources.

    prepare(log_dir) -> (container, test_description)
    evaluate(container, prediction, log_dir) -> evaluation dict
    cleanup(container) stops/removes only that task's container.
    A completed but unresolved evaluation has success=True and eval_result=
    'unresolved'; infrastructure failures have success=False and a failure stage.
    """
    if execution_spec is not None or evaluation_ledger is not None or lifecycle_callbacks is not None:
        from evaluation_adapter import EvaluationLifecycleAdapter, require
        require(execution_spec is not None and evaluation_ledger is not None, "ADAPTER_SPEC_REQUIRED")
        if lifecycle_callbacks is None:
            from downstream_lifecycle import DownstreamLifecycleCallbacks
            lifecycle_callbacks = DownstreamLifecycleCallbacks(entry, prepare=prepare, evaluate=evaluate,
                cleanup=cleanup, runner=runner, codex_config=codex_config, timeout=timeout, policy=policy,
                max_patch_bytes=max_patch_bytes, patch_filter=patch_filter,
                source_fingerprint=source_fingerprint, recorder=recorder)
        return EvaluationLifecycleAdapter(evaluation_ledger, lifecycle_callbacks).execute(execution_spec)
    instance_id = entry["instance_id"]
    if not instance_id or PurePosixPath(instance_id).name != instance_id or instance_id in (".", ".."):
        raise ValueError("Invalid task ID")
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = out_dir / f"{instance_id}.json"
    if prediction_path.exists():
        saved = json.loads(prediction_path.read_text())
        metadata_path = out_dir / "tasks" / instance_id / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        task_completed(recorder)
        return TaskResult(saved["success"], saved, metadata)
    log_dir = out_dir / "tasks" / instance_id
    log_dir.mkdir(parents=True, exist_ok=True)
    timing = dict.fromkeys(TIMING_FIELDS, 0.0)
    metadata = {
        "instance_id": instance_id, "backend": "codex_exec", "status": "failed",
        "stage": "dataset_image_preparation", "base_commit": entry["base_commit"],
        "language": entry.get("language"), "timing": timing,
        "timeout_seconds": timeout, "codex_calls": 0,
        "submitted_at_ns": time.time_ns(),
        "log_paths": {"codex": str(log_dir / "codex.log"),
                      "tests": str(log_dir / "test_output.txt")},
    }
    prediction = dict(instance_id=instance_id, model_name_or_path=model_name,
                      model_patch="", proposed_model_patches=[], success=False,
                      eval_result="incomplete", backend="codex_exec")
    container = None
    temporary = None
    prompt = ""
    started = mark = time.monotonic()
    stage = "dataset_image_preparation"
    usage_records = []
    readiness_abort = None
    metadata["status"] = "running"
    _atomic_write(log_dir / "metadata.json", json.dumps(metadata).encode())
    if recorder is not None:
        recorder.refresh()
    try:
        prepared = prepare(log_dir)
        container, test_description = prepared[:2]
        metadata["instance_image_cached"] = prepared[2] if len(prepared) > 2 else None
        metadata["docker"] = prepared[3] if len(prepared) > 3 else None
        metadata["container_name"] = getattr(container, "name", None)
        timing[stage] += time.monotonic() - mark
        stage = metadata["stage"] = "task_workspace_extraction"
        mark = time.monotonic()
        prepared_repository = inspect_prepared_repository(container, entry["base_commit"])
        metadata["prepared_repository"] = prepared_repository
        metadata["container_base_commit"] = prepared_repository["base_commit"]
        temporary = Path(tempfile.mkdtemp(prefix=".task-", dir=log_dir))
        workspace = temporary / "testbed"
        export_testbed(container, workspace)
        _git(workspace, "init", "-q", "--template=")
        _git(workspace, "add", "--all", "--force", ".")
        _git(workspace, "-c", "user.name=WGM task", "-c", "user.email=task@localhost",
             "commit", "-qm", "Task base state", "--allow-empty")
        baseline = _git(workspace, "rev-parse", "HEAD").decode().strip()
        git_config = (workspace / ".git/config").read_bytes()
        metadata["host_baseline"] = baseline
        metadata["patch_base"] = "prepared_repository"
        timing[stage] += time.monotonic() - mark
        stage = metadata["stage"] = "codex_execution"
        mark = time.monotonic()
        base = getattr(runner, "config", None) or codex_config or CodexExecConfig()
        if any(arg.split("=")[0] == "--add-dir" for arg in base.extra_args):
            raise TaskFailure("Downstream cannot grant additional writable directories")
        configured = replace(base, sandbox="workspace-write", approve_for_me=True,
                             ephemeral=True, output_schema=None, timeout_seconds=timeout)
        if runner is None:
            runner = CodexExecRunner(configured)
        else:
            runner = copy(runner)
            runner.config = configured
        prompt = task_prompt(policy, entry["problem_statement"], test_description)
        metadata["policy_sha256"] = hashlib.sha256(policy.encode()).hexdigest()
        receipt = await_runner(runner, recorder=recorder)
        if receipt is not None:
            mark = time.monotonic()  # Keep opt-in waiting out of model timing.
        metadata["codex_calls"] = 1
        execution = invoke(runner, CodexExecRequest(prompt, workspace, readiness_receipt=receipt), "downstream",
                           recorder=recorder, records=usage_records)
        metadata["execution"] = {
            "returncode": execution.returncode, "timed_out": execution.timed_out,
            "duration_seconds": execution.duration_seconds,
            "stderr_summary": _stderr_summary(execution.stderr_summary, prompt),
            "event_error_summary": _stderr_summary(execution.event_error_summary, prompt),
        }
        # Only a sanitized summary is persisted, never stdout/JSONL/source/prompt.
        summary = _stderr_summary(execution.last_message or "", prompt)
        cli_errors = "\n".join(filter(None, (metadata["execution"]["stderr_summary"],
                                             metadata["execution"]["event_error_summary"])))
        _atomic_write(log_dir / "codex.log", (cli_errors + "\n" + summary).encode())
        _atomic_write(out_dir / f"{instance_id}.md",
                      (summary + "\n" + cli_errors).encode())
        if not execution.ok:
            raise TaskFailure("Codex timed out" if execution.timed_out else
                              f"Codex exited with status {execution.returncode}")
        timing[stage] += time.monotonic() - mark
        stage = metadata["stage"] = "patch_generation_application"
        mark = time.monotonic()
        _check_baseline(workspace, baseline, git_config)
        if "language" in entry:  # Polyglot task
            removed = remove_untracked_artifacts(workspace)
            if removed:
                metadata["removed_untracked_artifacts"] = removed
        task_patch, paths = _capture_patch(workspace, baseline, max_patch_bytes)
        if patch_filter:
            task_patch = patch_filter(task_patch.decode()).encode()
            paths = _patch_paths(workspace, task_patch)
        patch_path = log_dir / "task_patch.diff"
        _atomic_write(patch_path, task_patch)
        metadata["patch_sha256"] = hashlib.sha256(task_patch).hexdigest()
        metadata["changed_files"] = paths
        metadata["patch_bytes"] = len(task_patch)
        put_bytes(container, "/tmp/wgm-task.patch", task_patch)
        checked = container_command(container, ["git", "apply", "--check", "/tmp/wgm-task.patch"], check=False)
        if checked.exit_code:
            raise TaskFailure("Task patch failed git apply --check")
        applied = container_command(container, ["git", "apply", "/tmp/wgm-task.patch"], check=False)
        if applied.exit_code:
            raise TaskFailure("Task patch could not be applied")
        prediction["model_patch"] = task_patch.decode()
        timing[stage] += time.monotonic() - mark
        stage = metadata["stage"] = "container_evaluation"
        mark = time.monotonic()
        evaluation = evaluate(container, prediction, log_dir)
        metadata["evaluation"] = evaluation
        if evaluation.get("timed_out") or evaluation.get("error"):
            raise TaskFailure("Container evaluation timed out" if evaluation.get("timed_out")
                              else "Container evaluation failed")
        prediction.update(success=True, eval_result="resolved" if evaluation["resolved"] else "unresolved")
        timing[stage] += time.monotonic() - mark
        stage = metadata["stage"] = "complete"
        metadata["status"] = "success"
    except ReadinessAbort as error:
        readiness_abort = error
        raise
    except (Exception, KeyboardInterrupt) as error:
        metadata["status"] = "failed"
        if stage in timing:
            timing[stage] += time.monotonic() - mark
        reason = str(error) if isinstance(error, (TaskFailure, CandidateError)) else "Task stage failed"
        if isinstance(error, KeyboardInterrupt):
            reason = "Task interrupted"
        metadata["safe_error"] = reason
        # Preserve actionable SDK/network errors without dumping a traceback,
        # source, prompts or credentials. Preparation can fail before prompt
        # assembly (and before a Docker logger exists).
        private_text = "\n".join((prompt, policy, entry.get("problem_statement", ""),
                                  entry.get("test_patch", ""), entry.get("patch", "")))
        metadata["error_type"] = type(error).__name__
        detail = str(error)
        if isinstance(error, ContainerCommandFailure):
            metadata["container_command"] = {
                "argv": [_stderr_summary(arg, private_text) for arg in error.argv],
                "returncode": error.returncode,
            }
            detail += "\ncommand: " + json.dumps(metadata["container_command"]["argv"])
        if stage == "codex_execution" and "execution" in metadata:
            detail += "\n" + "\n".join(filter(None, (
                metadata["execution"]["stderr_summary"], metadata["execution"]["event_error_summary"])))
        metadata["error_summary"] = _stderr_summary(detail, private_text)
        error_log = log_dir / "error.log"
        metadata["log_paths"]["error"] = str(error_log)
        _atomic_write(error_log, (
            f"{stage}: {metadata['error_type']}\n{metadata['error_summary']}\n"
        ).encode())
        if reason == "Patch is empty":
            prediction["eval_result"] = "empty_patch"
        else:
            prediction["eval_result"] = "error"
        metadata["interrupted"] = isinstance(error, KeyboardInterrupt)
        if metadata["interrupted"]:
            prediction["eval_result"] = "incomplete"
    finally:
        if readiness_abort is not None:
            # No prediction/failure binding for a gate cancellation. Cleanup must
            # not replace the controller stop with a scored origin failure.
            for resource, clean in ((temporary, shutil.rmtree), (container, cleanup)):
                if resource is not None:
                    try: clean(resource)
                    except BaseException: pass
        else:
            for resource, clean in ((temporary, shutil.rmtree), (container, cleanup)):
                if resource is not None:
                    try:
                        clean(resource)
                    except Exception:
                        metadata.update(status="failed", stage="cleanup", safe_error="Task resource cleanup failed")
                        prediction.update(success=False, eval_result="error")
                        if resource is container:
                            prediction.update(cleanup_failed=True, container_name=metadata.get("container_name"))
                        else:
                            metadata["retained_workspace"] = str(resource)
            # Preserve the existing diagnosis log discovery contract.
            test_log = log_dir / "test_output.txt"
            evaluation_log = test_log.read_bytes() if test_log.is_file() else metadata.get("safe_error", "").encode()
            _atomic_write(out_dir / f"{instance_id}_eval.md", evaluation_log)
            timing["per_task_total"] = time.monotonic() - started
            metadata["success"] = prediction["success"]
            metadata["codex_usage"] = aggregate_usage(usage_records)
            metadata["codex_call_records"] = usage_records
            mark = time.monotonic()
            _atomic_write(log_dir / "metadata.json", json.dumps(metadata, indent=2).encode())
            _atomic_write(prediction_path, json.dumps(prediction, indent=2).encode())
            timing["report_generation"] += time.monotonic() - mark
            timing["per_task_total"] = time.monotonic() - started
            _atomic_write(log_dir / "metadata.json", json.dumps(metadata, indent=2).encode())
            if recorder is not None:
                recorder.refresh()
    task_completed(recorder)
    if metadata.get("interrupted"):
        raise KeyboardInterrupt
    return TaskResult(prediction["success"], prediction, metadata)


def aggregate_report(predictions, total_instances):
    """Report keys consumed by get_all_performance and the existing harnesses."""
    report = {"total_instances": total_instances, "schema_version": 2}
    groups = {
        "submitted": [p["instance_id"] for p in predictions],
        "completed": [p["instance_id"] for p in predictions if p["success"]],
        "incomplete": [p["instance_id"] for p in predictions if p["eval_result"] == "incomplete"],
    }
    for kind in ("resolved", "unresolved", "empty_patch", "error"):
        groups[kind] = [p["instance_id"] for p in predictions if p["eval_result"] == kind]
    for name, ids in groups.items():
        report[f"{name}_ids"] = sorted(ids)
        report[f"{name}_instances"] = len(ids)
    leftovers = [p.get("container_name") or p["instance_id"] for p in predictions if p.get("cleanup_failed")]
    report.update(unstopped_instances=len(leftovers), unstopped_containers=leftovers, unremoved_images=[])
    return report


def select_entries(dataset, task_ids=None, n_tasks=None):
    """Keep requested order, fail on unknown IDs, and slice before model work."""
    by_id = {entry["instance_id"]: entry for entry in dataset}
    ids = list(dict.fromkeys(task_ids)) if task_ids is not None else list(by_id)
    if set(ids) - by_id.keys():
        raise ValueError("Requested task IDs are absent from the configured dataset")
    if n_tasks is not None:
        if isinstance(n_tasks, bool) or n_tasks < 1:
            raise ValueError("n_tasks must be positive")
        ids = ids[:n_tasks]
    return [by_id[task] for task in ids]


def load_agent_policy(source, model_patch_paths=(), *, polyglot=False):
    """Reconstruct the node, then read its evolvable instructions as data only.

    Never import/execute Python from the node, legacy modules or best_agent.
    Existing patch chains retain their ordering and files are never rewritten.
    """
    source = Path(source).resolve()
    if "best_agent" in source.parts:
        raise ValueError("best_agent is an archive, not a runtime agent source")
    if not model_patch_paths:
        return _read_policy(source / "prompts/downstream_instructions.txt")
    with tempfile.TemporaryDirectory(prefix="wgm-agent-policy-") as directory:
        workspace = Path(directory) / "agent"
        _copy_source(source, workspace, polyglot)
        _git(workspace, "init", "-q", "--template=")
        for path in model_patch_paths or ():
            content = Path(path).read_bytes()
            _patch_paths(workspace, content)
            _git(workspace, "apply", "--check", input=content)
            _git(workspace, "apply", input=content)
        return _read_policy(workspace / "prompts/downstream_instructions.txt")


def _read_policy(policy):
    if not policy.is_file() or policy.is_symlink():
        raise ValueError("Agent source needs prompts/downstream_instructions.txt; prepare a new Codex initial source")
    if policy.stat().st_size > 64 * 1024:
        raise ValueError("Downstream agent instructions exceed 64 KiB")
    text = policy.read_text()
    if not text.strip():
        raise ValueError("Downstream agent instructions are empty")
    return text


def load_swe_dataset(dataset_name, dataset_path):
    """Use the explicitly prepared dataset; a run never silently downloads data."""
    from datasets import load_from_disk
    path = Path(dataset_path)
    marker = path / "wgm_dataset.json"
    if not marker.is_file() or json.loads(marker.read_text()).get("dataset_name") != dataset_name:
        raise ValueError("SWE dataset is missing or mismatched; run python pipeline.py prepare-data --config config.yaml")
    return load_from_disk(str(path))


def run_serial_tasks(entries, out_dir, process, *, total_instances, configured_workers,
                     preparation_seconds=0.0):
    """Serial dispatch, atomic per-task results and a recoverable batch summary."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    previous_total = 0.0
    if (out_dir / "batch.json").is_file():
        previous_total = json.loads((out_dir / "batch.json").read_text())["timing"]["whole_run_total"]
    print(f"Downstream workers: configured={configured_workers}, effective=1, backend=codex_exec", flush=True)
    started = time.monotonic()
    records = []
    try:
        for index, entry in enumerate(entries, 1):
            print(f"Task {index}/{len(entries)}: {entry['instance_id']}", flush=True)
            result = process(entry, out_dir)
            records.append(result.prediction)
            print(f"Task {entry['instance_id']}: {result.prediction['eval_result']}", flush=True)
            if not result.success:
                detail = result.metadata.get("error_summary") or result.metadata.get("safe_error", "")
                print(f"  Failed at {result.metadata['stage']} "
                      f"({result.metadata.get('error_type', 'TaskFailure')}): {detail}", flush=True)
    finally:
        # Use committed prediction files, including tasks finished before resume.
        predictions = [json.loads(path.read_text()) for path in out_dir.glob("*.json")
                       if path.name != "batch.json"]
        predictions = [p for p in predictions if p.get("backend") == "codex_exec"]
        task_metadata = [json.loads(p.read_text()) for p in (out_dir / "tasks").glob("*/metadata.json")]
        timing = {
            "dataset_preparation": preparation_seconds,
            "whole_run_total": previous_total + preparation_seconds + time.monotonic() - started,
            "configured_workers": configured_workers, "effective_workers": 1,
            "tasks": [{"instance_id": m["instance_id"], **m["timing"],
                       "instance_image_cached": m.get("instance_image_cached"),
                       "timeout_seconds": m["timeout_seconds"]} for m in task_metadata],
        }
        _atomic_write(out_dir / "batch.json", json.dumps({
            "timing": timing, "report": aggregate_report(predictions, total_instances),
        }, indent=2).encode())
    return [out_dir]
