"""One isolated host-side self-improvement attempt; no Docker or benchmark imports."""

from copy import copy
from dataclasses import dataclass, field, replace
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import subprocess
import tempfile
import time
from network_readiness import ReadinessAbort, await_runner

from provider_prompts import policy_note, provider_of
from runners import CodexExecConfig, CodexExecRequest, CodexExecRunner
from runners.codex_exec import _clean_environment, _execute, _stderr_summary, _validate_timeout
from runners.usage import invoke, aggregate_usage


@dataclass
class CandidateConfig:
    max_patch_bytes: int = 1024 * 1024
    keep_failed_workspace: bool = False
    # Explicit, trusted local test command. Empty means syntax checks only;
    # never discover or run a benchmark automatically.
    validation_command: list[str] = field(default_factory=list)
    validation_timeout_seconds: float = 120.0


@dataclass(frozen=True)
class CandidateResult:
    ok: bool
    run_id: str
    patch_path: Path | None
    metadata: dict = field(repr=False)


class CandidateError(RuntimeError):
    """Only fixed, non-sensitive messages are raised by candidate checks."""


_CACHES = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
_EXCLUDED = _CACHES | {
    "best_agent",
    ".git", ".codex", ".ssh", ".aws", ".azure", ".config", ".cache",
    ".venv", "venv", "node_modules", "cache", "caches", "output", "outputs",
    "result", "results", "logs", "predictions", "initial_swe", "initial_polyglot",
    "swe-bench", "polyglot-benchmark", "datasets", "checkpoints",
}
_PRIVATE_FILES = {
    ".env", ".envrc", ".netrc", ".npmrc", ".pypirc", ".git-credentials", "auth.json",
    "credentials", "credentials.json", "credentials.toml", "id_rsa", "id_ed25519",
    "model_patch.diff", "metadata.json", "self_evo.md",
}


def _forbidden(path):
    parts = PurePosixPath(path).parts
    return (not parts or PurePosixPath(path).is_absolute() or ".." in parts or
            any(part.lower() in _EXCLUDED or part.lower() in _PRIVATE_FILES or
                part.lower().startswith((".env.", "output_", "result_", ".candidate-")) or
                part.lower().endswith((".pem", ".key", ".log", ".pyc"))
                for part in parts))


def _git(directory, *args, input=None):
    # Ignore inherited Git redirection and personal config: never touch the
    # caller's index/worktree, execute hooks, sign commits or run external diffs.
    environment = {k: v for k, v in _clean_environment().items() if not k.startswith("GIT_")}
    environment.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
                       GIT_TERMINAL_PROMPT="0")
    result = subprocess.run(
        ["git", "-C", str(directory), "-c", "core.hooksPath=" + os.devnull,
         "-c", "core.fsmonitor=false", "-c", "commit.gpgsign=false",
         "-c", "core.autocrlf=false", *args],
        input=input, capture_output=True, env=environment, timeout=30,
    )
    if result.returncode:
        raise CandidateError("Git operation failed")
    return result.stdout


def _atomic_write(path, data):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".candidate-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _copy_source(source, workspace, polyglot, *, preserve_source=False):
    """Copy replayable files; V2 preserves its already-frozen source identity.

    The default retains the historical SWE/Polyglot platform conversion.
    That conversion must not run on a V2 parent already bound by its plan.
    """
    def ignore(directory, names):
        relative = Path(directory).relative_to(source)
        return [name for name in names if _forbidden((relative / name).as_posix()) or
                (Path(directory) / name).is_symlink() or
                ((Path(directory) / name).is_file() and
                 (Path(directory) / name).stat().st_size > 20 * 1024 * 1024)]
    shutil.copytree(source, workspace, ignore=ignore)
    if preserve_source:
        return
    # Match the legacy /hgm view used to create AND consume the patch chain.
    variant = workspace / "coding_agent_polyglot.py"
    if polyglot:
        if not variant.is_file():
            raise CandidateError("Initial Polyglot agent source is missing")
        variant.replace(workspace / "coding_agent.py")
        for name in ("utils/eval_utils.py", "utils/swe_log_parsers.py"):
            (workspace / name).unlink(missing_ok=True)
    else:
        variant.unlink(missing_ok=True)


def _patch_paths(directory, patch):
    # --no-renames on generated patches keeps this format compatible with the
    # evaluators' `patch -p1`; numstat -z also handles spaces and quoted names.
    if not patch.strip():
        raise CandidateError("Patch is empty")
    records = _git(directory, "apply", "--numstat", "-z", "-p1", input=patch)
    paths = []
    for record in records.split(b"\0"):
        if not record:
            continue
        added, removed, name = record.split(b"\t", 2)
        if added == b"-" or removed == b"-":
            raise CandidateError("Binary patches are not supported")
        path = name.decode("utf-8")
        if _forbidden(path):
            raise CandidateError("Patch contains a forbidden path")
        paths.append(path)
    if not paths:
        raise CandidateError("Patch is empty")
    return paths


def _check_workspace(workspace):
    # Never follow links out of the candidate. Routine Python test caches may
    # exist locally but are excluded from staging; other forbidden files fail.
    for directory, dirs, files in os.walk(workspace, followlinks=False):
        base = Path(directory)
        dirs[:] = [name for name in dirs if name not in _CACHES and
                   not (base == workspace and name == ".git")]
        for name in dirs + files:
            path = base / name
            relative = path.relative_to(workspace).as_posix()
            if path.is_symlink() or _forbidden(relative):
                raise CandidateError("Candidate contains a forbidden path or symlink")


def _check_baseline(workspace, baseline, git_config):
    if (any((workspace / name).is_symlink() for name in (".git", ".git/config", ".git/HEAD")) or
            not (workspace / ".git").is_dir() or
            (workspace / ".git/config").read_bytes() != git_config or
            _git(workspace, "rev-parse", "HEAD").decode().strip() != baseline):
        raise CandidateError("Candidate changed Git baseline or management files")


def generate_candidate(src_path, run_dir, parent_patch_paths, problem_statement,
                       test_description, *, parent_commit="initial", entry=None,
                       polyglot=False, codex_config=None, config=None, runner=None, recorder=None):
    """Generate one incremental patch, with no retry or evaluator invocation.

    run_dir is owned by this attempt; an existing model_patch.diff is never
    overwritten. Only temporary repositories inside run_dir receive commits.
    A failed result is never an eligible child. Tests are authoritative only
    when validation_command is configured; model prose is not a test verdict.
    """
    from candidate_provenance import (ROOT, ProvenanceError, read_chain, replay,
                                      source_digest, with_provenance)
    config = config if config is not None else CandidateConfig()
    run_dir = Path(run_dir).resolve()
    metadata = {
        "candidate_schema_version": 1, "artifact_type": "candidate_attempt", "provenance": None,
        "run_id": run_dir.name, "parent_commit": parent_commit, "entry": entry,
        "backend": "codex_exec", "status": "failed", "stage": "workspace",
        "overall_performance": {
            "accuracy_score": 0.0, "total_resolved_instances": 0,
            "total_submitted_instances": 0, "files": [], "total_submitted_ids": [],
            "total_unresolved_ids": [], "total_emptypatch_ids": [], "total_resolved_ids": [],
        },
    }
    started = time.monotonic()
    temporary = None
    prompt = ""
    patch_path = run_dir / "model_patch.diff"
    published = False
    owns_output = False
    usage_records = []
    readiness_abort = None
    try:
        source = Path(src_path).resolve()
        if run_dir.is_relative_to(source) or source.is_relative_to(run_dir):
            raise CandidateError("Candidate output must be outside initial source")
        run_dir.mkdir(parents=True, exist_ok=True)
        if (patch_path.exists() or patch_path.is_symlink() or
                (run_dir / "metadata.json").exists() or (run_dir / "metadata.json").is_symlink()):
            raise CandidateError("Candidate patch already exists")
        owns_output = True
        if not source.is_dir():
            raise CandidateError("Initial source directory is missing")
        metadata["stage"] = "configuration"
        if (isinstance(config.max_patch_bytes, bool) or not isinstance(config.max_patch_bytes, int)
                or config.max_patch_bytes <= 0 or not isinstance(config.keep_failed_workspace, bool)):
            raise CandidateError("Invalid candidate limits")
        if (not isinstance(config.validation_command, list) or
                any(not isinstance(arg, str) or not arg for arg in config.validation_command)):
            raise CandidateError("Invalid validation command")
        _validate_timeout(config.validation_timeout_seconds)
        base_config = getattr(runner, "config", None) if runner is not None else codex_config
        base_config = base_config if isinstance(base_config, CodexExecConfig) else CodexExecConfig()
        if any(arg.split("=")[0] == "--add-dir" for arg in base_config.extra_args):
            raise CandidateError("Candidate runner cannot grant additional writable directories")
        execution_config = replace(base_config, sandbox="workspace-write", approve_for_me=True,
                                   ephemeral=True, output_schema=None)
        if runner is None:
            runner = CodexExecRunner(execution_config)
        elif isinstance(getattr(runner, "config", None), CodexExecConfig):
            runner = copy(runner)
            runner.config = execution_config

        metadata["stage"] = "workspace"
        temporary = Path(tempfile.mkdtemp(prefix=".candidate-", dir=run_dir))
        workspace, parent = temporary / "workspace", temporary / "parent"
        _copy_source(source, workspace, polyglot)
        _git(workspace, "init", "-q", "--template=")
        metadata["stage"] = "parent_patches"
        parent_chain = read_chain(run_dir.parent, parent_commit)
        expected_paths = [run_dir.parent / p.provenance.candidate_id / "model_patch.diff"
                          for p in parent_chain]
        if [Path(p).absolute() for p in parent_patch_paths] != expected_paths:
            raise ProvenanceError("Supplied parent patches differ from validated parent chain", code="PROV_PARENT_PATHS")
        parent_provenance = parent_chain[-1].provenance if parent_chain else ROOT
        replay(workspace, parent_chain)
        metadata["ancestor_patch_sha256"] = [p.patch_sha256 for p in parent_provenance.complete_chain]
        _check_workspace(workspace)
        metadata["parent_patch_count"] = len(parent_patch_paths)
        parent_source = source_digest(workspace)

        metadata["stage"] = "baseline"
        _git(workspace, "add", "--all", "--force", "--", ".")
        _git(workspace, "-c", "user.name=WGM candidate", "-c", "user.email=candidate@localhost",
             "commit", "-q", "--allow-empty", "-m", "Parent state baseline")
        baseline = _git(workspace, "rev-parse", "HEAD").decode().strip()
        git_config = (workspace / ".git/config").read_bytes()
        metadata["baseline"] = baseline
        # Keep the verifier outside Codex's workspace and reconstruct the exact
        # parent state before execution, including the applied ancestor patches.
        shutil.copytree(workspace, parent, ignore=shutil.ignore_patterns(".git"))
        _git(parent, "init", "-q", "--template=")
        prompt = (
            f"Modify only the candidate workspace {workspace} to solve this self-improvement problem.\n"
            f"<problem_description>\n{problem_statement}\n</problem_description>\n\n"
            f"<test_description>\n{test_description.replace('/hgm/', shlex.quote(str(workspace) + '/'))}\n</test_description>\n\n"
            "You may inspect files and run lightweight tests. Do not invoke a model, Docker, "
            "SWE-bench or Polyglot evaluations. Do not install dependencies. "
            "Do not modify benchmark data, outputs, credentials "
            "or Git management files. Do not create commits. Avoid unrelated changes. "
            "The result is collected from Git diff, including new and deleted files; "
            "do not return a textual patch. Briefly report the changes and tests."
        )
        if (workspace / "prompts/downstream_instructions.txt").is_file():
            prompt += policy_note(provider_of(execution_config))
        metadata["problem_sha256"] = hashlib.sha256(problem_statement.encode()).hexdigest()
        metadata["stage"] = "codex"
        receipt = await_runner(runner, recorder=recorder)
        result = invoke(runner, CodexExecRequest(prompt, workspace, readiness_receipt=receipt), "self_improvement",
                        recorder=recorder, records=usage_records)
        metadata["execution"] = {
            "returncode": result.returncode, "timed_out": result.timed_out,
            "duration_seconds": result.duration_seconds,
            "stderr_summary": _stderr_summary(result.stderr_summary, prompt),
            "last_message": _stderr_summary(result.last_message or "", prompt),
        }
        if not result.ok:
            raise CandidateError("Codex timed out" if result.timed_out else "Codex exited unsuccessfully")

        metadata["stage"] = "validation"
        _check_baseline(workspace, baseline, git_config)
        _check_workspace(workspace)
        if config.validation_command:
            code, expired, stdout, stderr = _execute(
                config.validation_command, workspace, _clean_environment(), b"",
                config.validation_timeout_seconds,
            )
            metadata["validation"] = {
                "returncode": code, "timed_out": expired,
                "summary": _stderr_summary((stdout + stderr).decode("utf-8", errors="replace"), prompt),
            }
            if code != 0 or expired:
                raise CandidateError("Candidate tests timed out" if expired else "Candidate tests failed")
        else:
            metadata["validation"] = {"tests": "not_configured"}

        metadata["stage"] = "patch"
        _check_baseline(workspace, baseline, git_config)
        _check_workspace(workspace)
        # Explicit glob magic makes **/ match zero directories too, so caches
        # at the workspace root are excluded as well as nested test caches.
        excludes = [f":(glob,exclude)**/{name}/**" for name in sorted(_CACHES)]
        _git(workspace, "add", "--all", "--force", "--", ".", *excludes)
        # Codex may already have staged generated caches; add's exclusions do
        # not remove existing index entries, so filter the exported diff too.
        patch = _git(workspace, "diff", "--cached", "--no-ext-diff", "--no-textconv",
                     "--no-renames", "--src-prefix=a/", "--dst-prefix=b/", baseline, "--",
                     ".", *excludes)
        if len(patch) > config.max_patch_bytes:
            raise CandidateError("Patch exceeds configured size limit")
        paths = _patch_paths(workspace, patch)
        for name in paths:
            path = workspace / name
            if path.suffix == ".py" and path.is_file():
                compile(path.read_bytes(), name, "exec")
        metadata["validation"]["python_syntax"] = "passed"
        _git(parent, "apply", "--check", "-p1", input=patch)
        metadata["patch_bytes"] = len(patch)
        metadata["patch_sha256"] = hashlib.sha256(patch).hexdigest()
        metadata["changed_files"] = paths
        metadata["stage"] = "publish"
        published_metadata = with_provenance(
            {**metadata, "status": "success", "stage": "complete"}, run_dir.name,
            parent_provenance, patch, parent_source, source_digest(workspace))
        _atomic_write(patch_path, patch)
        published = True
        metadata = published_metadata
    except ReadinessAbort as error:
        readiness_abort = error
        raise
    except Exception as error:
        metadata["reason"] = str(error) if isinstance(error, (CandidateError, ProvenanceError)) else "Candidate stage failed"
        if isinstance(error, ProvenanceError):
            metadata["reason_code"] = error.code
    finally:
        if readiness_abort is not None:
            # The selected candidate identity survives; no failed candidate is published.
            if temporary is not None:
                try: shutil.rmtree(temporary)
                except BaseException: pass
        else:
            metadata["codex_usage"] = aggregate_usage(usage_records)
            metadata["codex_call_records"] = usage_records
            if temporary is not None:
                if metadata["status"] == "failed" and config.keep_failed_workspace:
                    metadata["retained_workspace"] = str(temporary / "workspace")
                else:
                    try:
                        shutil.rmtree(temporary)
                    except OSError:
                        metadata.update(status="failed", stage="cleanup", reason="Workspace cleanup failed")
                        metadata["retained_workspace"] = str(temporary / "workspace")
            metadata["duration_seconds"] = time.monotonic() - started
            if metadata["status"] == "failed" and published:
                patch_path.unlink(missing_ok=True)
            if metadata["status"] == "failed":
                metadata["artifact_type"] = "candidate_attempt"
            try:
                if owns_output:
                    log_text = (f"status={metadata['status']} stage={metadata['stage']}\n"
                                f"reason={metadata.get('reason', '')}\n" +
                                json.dumps(metadata.get("execution", {}), indent=2))
                    _atomic_write(run_dir / "candidate.log", log_text.encode())
                    _atomic_write(run_dir / "metadata.json", json.dumps(metadata, indent=2).encode())
            except OSError:
                metadata.update(status="failed", stage="metadata", reason="Could not save candidate metadata")
                if published:
                    patch_path.unlink(missing_ok=True)
                try:
                    _atomic_write(run_dir / "candidate.log", b"status=failed stage=metadata\nCould not save candidate metadata\n")
                except OSError:
                    pass
    ok = metadata["status"] == "success"
    return CandidateResult(ok, run_dir.name, patch_path if ok else None, metadata)
