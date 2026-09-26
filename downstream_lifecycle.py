"""Production callbacks for the opt-in ledger path of downstream_solver.run_task."""
from copy import copy
from dataclasses import asdict, replace
import hashlib
from pathlib import Path
import shutil
import time
import tempfile

import downstream_solver as ds
import evaluation_ledger as el
from evaluation_adapter import (AdapterError, ExecutionEvidence, PreparedEvaluation, ModelExecutionResult,
    SealedModelArtifacts, EvaluatorExecutionResult, require, validate_preparation_context)
from runners import ClaudeCodeRunner, CodexExecConfig, CodexExecRequest, CodexExecRunner
from runners.codex_exec import CodexConfigurationError, CodexStartError
from runners.usage import invoke
from network_readiness import await_runner, enabled as readiness_enabled, ReadinessAbort

# Transient network failures (e.g. an SSH-forwarded proxy reconnecting) are
# retried every PREPARATION_RETRY_SECONDS for at most PREPARATION_NETWORK_WAIT_SECONDS.
PREPARATION_NETWORK_WAIT_SECONDS = 3600.0
PREPARATION_RETRY_SECONDS = 30.0


def _transient_network_error(error):
    """Connection, proxy and timeout failures of HTTP dependency downloads."""
    names = {"ConnectionError", "ProxyError", "Timeout", "ConnectTimeout", "ReadTimeout",
             "NewConnectionError", "MaxRetryError", "ProtocolError", "RemoteDisconnected"}
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        module = type(error).__module__ or ""
        if type(error).__name__ in names and module.split(".")[0] in ("requests", "urllib3", "http"):
            return True
        error = error.__cause__ or error.__context__
    return False


def sha(data):
    return hashlib.sha256(data).hexdigest()


def workspace_fingerprint(path):
    rows = []
    for p in sorted(path.rglob("*")):
        relative = p.relative_to(path)
        if ".git" in relative.parts or any(part in ds._CACHES for part in relative.parts):
            continue
        if p.is_symlink():
            require(p.resolve().is_relative_to(path), "ADAPTER_WORKSPACE_INVALID")
            rows.append([relative.as_posix(), "link", str(p.readlink())])
        elif p.is_file():
            rows.append([relative.as_posix(), "file", sha(el._regular(p).read_bytes())])
    return el.digest(rows)


class DownstreamLifecycleCallbacks:
    def __init__(self, entry, *, prepare, evaluate, cleanup, runner=None, codex_config=None,
                 timeout=1800, policy="", max_patch_bytes=1024*1024, patch_filter=None,
                 source_fingerprint=None, recorder=None):
        self.entry = dict(entry)
        self.prepare_callback, self.evaluate_callback, self.cleanup_callback = prepare, evaluate, cleanup
        self.policy, self.patch_filter, self.max_patch_bytes = policy, patch_filter, max_patch_bytes
        self.source_fingerprint, self.recorder = source_fingerprint, recorder
        base = getattr(runner, "config", None) or codex_config or CodexExecConfig()
        require(not any(a.split("=")[0] == "--add-dir" for a in base.extra_args), "ADAPTER_CONFIG_INVALID")
        self.config = replace(base, sandbox="workspace-write", approve_for_me=True,
                              ephemeral=True, output_schema=None, timeout_seconds=timeout)
        self.runner = copy(runner) if runner is not None else CodexExecRunner(self.config)
        self.runner.config = self.config
        if readiness_enabled(self.config) and runner is not None:
            # A copied runner must not inherit another worker's receipt authority.
            from network_readiness import ReadinessGate
            previous = getattr(runner, 'readiness_gate', None)
            self.runner.readiness_gate = (ReadinessGate(self.config.network_readiness,
                check=previous.check, monotonic=previous.monotonic, wall=previous.wall,
                waiter=previous.waiter, log=previous.log, cancel_event=self.config.readiness_cancel_event)
                if previous is not None else None)
        self.container = None
        self.description = None
        self.preparation_attempted = False
        self.resource_owner = None
        self.owned_workdir = None
        self.readiness_receipt = None

    def _preparation_context(self, spec):
        return (spec.preparation_envelope() if readiness_enabled(self.config) or spec.attempt_id != 'attempt-000001'
                else spec.envelope())

    def await_model_readiness(self):
        self.readiness_receipt = await_runner(self.runner, recorder=self.recorder)

    def validate_model_readiness(self):
        if readiness_enabled(self.config):
            self._check(self.resource_owner)
            gate = self.runner.readiness_gate
            if gate.authority.rejection(self.readiness_receipt, gate.config, gate.monotonic()) == 'EXPIRED':
                return False  # Still before the marker: the adapter must re-read state and wait again.
        await_runner(self.runner, self.readiness_receipt, reject_stale=True)
        return True

    def _check(self, spec):
        i = spec.identity
        require(self.entry["instance_id"] == i.task_id and self.entry["base_commit"] == spec.base_commit,
                "ADAPTER_IDENTITY_MISMATCH")
        require(el.digest(self.entry) == i.task_input_fingerprint and sha(self.policy.encode()) == i.policy_fingerprint
                and el.digest(asdict(self.config)) == i.model_fingerprint
                and self.source_fingerprint == i.source_fingerprint
                and getattr(self.evaluate_callback, "evaluator_fingerprint", None) == i.evaluator_fingerprint,
                "ADAPTER_IDENTITY_MISMATCH")
        require(callable(getattr(self.evaluate_callback, "lifecycle", None)), "ADAPTER_EVALUATOR_UNSUPPORTED")

    def prepare(self, spec, workdir):
        self._check(spec)  # Before any preparation subprocess/container.
        require(self.resource_owner is None and self.container is None, "ADAPTER_RESOURCE_MISMATCH")
        self.owned_workdir = el._checked_path(workdir)
        self.preparation_attempted = True
        prepared, retries = self._prepare_with_network_retry(workdir)
        self.container, self.description = prepared[:2]
        resource_id = getattr(self.container, "id", None)
        require(type(resource_id) is str, "ADAPTER_RESOURCE_IDENTITY_MISSING")
        el._id(resource_id)
        self.resource_owner = spec
        el._publish(workdir / "resource.json", {"context":self._preparation_context(spec), "resource_id":resource_id})
        repository = ds.inspect_prepared_repository(self.container, spec.base_commit)
        require(repository["base_commit"] == spec.base_commit, "ADAPTER_SOURCE_MISMATCH")
        workspace = workdir / "testbed"
        ds.export_testbed(self.container, workspace)
        ds._git(workspace, "init", "-q", "--template=")
        ds._git(workspace, "add", "--all", "--force", ".")
        ds._git(workspace, "-c", "user.name=WGM task", "-c", "user.email=task@localhost",
                "commit", "-qm", "Task base state", "--allow-empty")
        baseline = ds._git(workspace, "rev-parse", "HEAD").decode().strip()
        testspec = getattr(self.prepare_callback, "testspec_fingerprint", None)
        image = prepared[3] if len(prepared) > 3 else None
        require(testspec is not None and image is not None, "ADAPTER_PREPARATION_IDENTITY_MISSING")
        result = PreparedEvaluation(spec, workspace_fingerprint(workspace), self.source_fingerprint,
                                    testspec, el.digest(image), baseline,
                                    sha((workspace / ".git/config").read_bytes()), resource_id)
        row = asdict(result); row.pop("spec")
        if retries:
            row["network_retries"] = retries  # Absent when preparation succeeded first time.
        el._publish(workdir / "prepared.json", {"context":self._preparation_context(spec), **row})
        return result

    def _prepare_with_network_retry(self, workdir):
        """Retry transient network failures of image/dependency preparation.

        Only connection/proxy/timeout errors are retried, every
        PREPARATION_RETRY_SECONDS until PREPARATION_NETWORK_WAIT_SECONDS have
        passed: they occur while downloading dependencies, before any
        container, model call or evaluation exists. Every other failure
        propagates unchanged on the first attempt.
        """
        retries, deadline = [], time.monotonic() + PREPARATION_NETWORK_WAIT_SECONDS
        while True:
            try:
                return self.prepare_callback(workdir), retries
            except Exception as error:
                if not _transient_network_error(error) or time.monotonic() >= deadline:
                    raise
                retries.append({"exception_type": type(error).__name__, "at_ns": time.time_ns()})
                time.sleep(PREPARATION_RETRY_SECONDS)

    def _resource(self, spec, workdir, *, cleanup=False, resource_id=None):
        workdir = el._checked_path(workdir)
        require(self.owned_workdir is None or self.owned_workdir == workdir, "ADAPTER_RESOURCE_MISMATCH")
        require(self.resource_owner is None or self.resource_owner == spec, "ADAPTER_RESOURCE_MISMATCH")
        row = (el._read_json(workdir / "resource.json") if (workdir / "resource.json").exists()
               else {"context":self._preparation_context(spec), "resource_id":resource_id})
        el._keys(row, ("context", "resource_id"))
        validate_preparation_context(spec, row.get("context"))
        require(row.get("resource_id") is not None, "ADAPTER_PREPARATION_UNKNOWN")
        el._id(row["resource_id"])
        require(resource_id is None or row["resource_id"] == resource_id, "ADAPTER_RESOURCE_MISMATCH")
        if self.container is None:
            restore = getattr(self.prepare_callback, "restore", None)
            require(callable(restore), "ADAPTER_RESOURCE_RESTORE_UNSUPPORTED")
            self.container, self.description = restore(row["resource_id"], workdir, cleanup=cleanup)
        require(getattr(self.container, "id", None) == row["resource_id"], "ADAPTER_RESOURCE_MISMATCH")
        self.resource_owner, self.owned_workdir = spec, workdir

    def recover_preparation(self, spec, workdir):
        self._check(spec)
        # No proof of ownership means an unknown preparation, not a fresh start.
        require((workdir / "resource.json").is_file(), "ADAPTER_PREPARATION_UNKNOWN")
        if not (workdir / "prepared.json").is_file():
            # A fresh controller cannot corroborate a partial resource record.
            # Only the original in-memory owner may clean incomplete setup.
            require(self.resource_owner == spec and self.container is not None,
                    "ADAPTER_PREPARATION_UNKNOWN")
            self.cleanup(spec, workdir)
            raise AdapterError("ADAPTER_PREPARATION_INCOMPLETE")
        data = el._read_json(workdir / "prepared.json")
        validate_preparation_context(spec, data.pop("context", None))
        prepared = PreparedEvaluation(spec, **data)
        # Validate both identities before any Docker lookup or cleanup.
        require(workspace_fingerprint(workdir / "testbed") == prepared.workspace_fingerprint,
                "ADAPTER_WORKSPACE_INVALID")
        self.restore(prepared, workdir, for_model=True)
        return prepared

    def restore(self, prepared, workdir, *, for_model=False):
        self._check(prepared.spec)
        self._resource(prepared.spec, workdir, resource_id=prepared.resource_id)
        from dgm_recovery import resume_resource
        resume_resource(self.container, prepared.spec, prepared, workdir)
        repository = ds.inspect_prepared_repository(self.container, prepared.spec.base_commit)
        require(repository["base_commit"] == prepared.spec.base_commit, "ADAPTER_SOURCE_MISMATCH")
        require(getattr(self.prepare_callback, "testspec_fingerprint", None) == prepared.testspec_fingerprint,
                "ADAPTER_TESTSPEC_MISMATCH")
        require(getattr(self.prepare_callback, "image_fingerprint", None) == prepared.image_fingerprint,
                "ADAPTER_IMAGE_MISMATCH")
        if for_model:
            workspace = el._checked_path(workdir / "testbed")
            config = el._regular(workspace / ".git/config").read_bytes()
            require(sha(config) == prepared.git_config_sha256, "ADAPTER_WORKSPACE_INVALID")
            ds._check_baseline(workspace, prepared.baseline, config)
            require(workspace_fingerprint(workspace) == prepared.workspace_fingerprint, "ADAPTER_WORKSPACE_INVALID")

    def model(self, prepared, workdir, on_started):
        spec = prepared.spec
        require(self.description is not None, "ADAPTER_PREPARATION_UNKNOWN")
        # Same instructions and generation inputs as the ordinary single-task path.
        prompt = ds.task_prompt(self.policy, self.entry["problem_statement"], self.description)
        terminals, records = [], []
        started = time.monotonic()
        result, error = None, None
        try:
            result = invoke(self.runner, CodexExecRequest(prompt, workdir / "testbed", on_started,
                            spec.model_call_id, terminals.append, self.readiness_receipt), "downstream", records=records,
                            recorder=self.recorder)
        except (Exception, KeyboardInterrupt, ReadinessAbort) as caught:
            error = caught
            # Typed configuration/start failures occur before Popen. Other
            # exceptions require the explicit runner callback as positive proof.
            if (not terminals and type(self.runner) in (CodexExecRunner, ClaudeCodeRunner)
                    and isinstance(caught, (CodexStartError, CodexConfigurationError))
                    and not any(r.get("started_calls") for r in records)):
                from runners.codex_exec import ProcessTerminal
                terminals.append(ProcessTerminal(False, False, None, launch_failed=True))
        usage = records[-1] if records else {}
        require(not records or all(r["call_id"] == spec.model_call_id for r in records), "ADAPTER_CALL_ID_MISMATCH")
        if not terminals:
            evidence = ExecutionEvidence(bool(usage.get("started_calls")), False, None,
                                         duration_seconds=time.monotonic()-started,
                                         usage_error_code=usage.get("usage_error_code"))
        else:
            t = terminals[-1]
            evidence = ExecutionEvidence(t.started, t.reaped, t.returncode, t.launch_failed,
                t.timed_out, t.interrupted, time.monotonic()-started,
                *[usage.get(k) for k in ("input_tokens", "cached_input_tokens", "output_tokens")],
                usage_error_code=usage.get("usage_error_code"))
        result = ModelExecutionResult(spec, spec.model_call_id, evidence, "MODEL_FAILED" if error else None)
        if isinstance(error, ReadinessAbort):
            error.model_result = result
            raise error
        return result

    def seal_model(self, prepared, workdir):
        self._check(prepared.spec)
        workspace = workdir / "testbed"
        config = el._regular(workspace / ".git/config").read_bytes()
        require(sha(config) == prepared.git_config_sha256, "ADAPTER_WORKSPACE_INVALID")
        ds._check_baseline(workspace, prepared.baseline, config)
        if "language" in self.entry:  # Polyglot task: same cleanup as the baseline path.
            removed = ds.remove_untracked_artifacts(workspace)
            if removed:
                el._publish(workdir / "removed_artifacts.json", {"removed_untracked_artifacts": removed})
        try:
            patch, _ = ds._capture_patch(workspace, prepared.baseline, self.max_patch_bytes)
        except ds.CandidateError as error:
            if str(error) != "Patch is empty":
                raise AdapterError("ADAPTER_PATCH_INVALID") from None
            patch = b""
        if self.patch_filter and patch:
            patch = self.patch_filter(patch.decode()).encode()
        if patch:
            ds._patch_paths(workspace, patch)
        return SealedModelArtifacts(prepared.spec, patch)

    def evaluate(self, prepared, sealed, workdir, on_started):
        spec = prepared.spec
        self._resource(spec, workdir, resource_id=prepared.resource_id)
        # The adapter's evaluator marker covers patch application and grading.
        try:
            if sealed.patch:
                ds.put_bytes(self.container, "/tmp/wgm-task.patch", sealed.patch)
                ds.container_command(self.container, ["git", "apply", "--check", "/tmp/wgm-task.patch"])
                ds.container_command(self.container, ["git", "apply", "/tmp/wgm-task.patch"])
        except ds.ContainerCommandFailure:
            return EvaluatorExecutionResult(spec, spec.evaluator_call_id,
                ExecutionEvidence(False, False, None, launch_failed=True), b"{}", "EVALUATOR_FAILED")
        prediction = {"instance_id":spec.identity.task_id, "model_name_or_path":spec.identity.candidate_id,
                      "model_patch":sealed.patch.decode(), "evaluation_id":spec.identity.evaluation_id,
                      "model_call_id":spec.model_call_id, "evaluator_call_id":spec.evaluator_call_id}
        raw = self.evaluate_callback.lifecycle(self.container, prediction, workdir, on_started=on_started)
        terminal = raw["terminal"]
        evidence = ExecutionEvidence(terminal.started, terminal.reaped, terminal.returncode,
            terminal.launch_failed, terminal.timed_out, terminal.interrupted, raw["duration_seconds"])
        report = {"instance_id":raw.get("instance_id"), "resolved":raw.get("resolved"),
                  "status":"completed", "error":None}
        return EvaluatorExecutionResult(spec, spec.evaluator_call_id, evidence, el.canonical(report), raw.get("error_code"))

    def cleanup(self, spec, workdir, *, prepared=None):
        workdir = el._checked_path(workdir)
        require(self.owned_workdir is None or self.owned_workdir == workdir, "ADAPTER_RESOURCE_MISMATCH")
        require(self.resource_owner is None or self.resource_owner == spec, "ADAPTER_RESOURCE_MISMATCH")
        if prepared is not None:
            require(prepared.spec == spec, "ADAPTER_RESOURCE_MISMATCH")
        resource_id = prepared.resource_id if prepared else (
            getattr(self.container, "id", None) if self.resource_owner == spec else None)
        if self.container is not None or (workdir / "resource.json").exists() or prepared is not None:
            self._resource(spec, workdir, cleanup=True, resource_id=resource_id)
        if self.container is None and self.preparation_attempted:
            require(getattr(self.prepare_callback, "cleanup_verified", False), "ADAPTER_CLEANUP_UNVERIFIED")
        if self.container is not None:
            self.cleanup_callback(self.container)
            self.container = None
        if workdir.exists():
            require(self.owned_workdir == workdir or not any(workdir.iterdir()), "ADAPTER_CLEANUP_UNVERIFIED")
            shutil.rmtree(el._checked_path(workdir))
        self.resource_owner, self.owned_workdir, self.preparation_attempted = None, None, False


def _isolate_evaluator(container):
    """Best effort stop of this attempt's verified owned container; no removal."""
    try:
        container.kill()
    except (Exception, KeyboardInterrupt):
        try:
            container.stop(timeout=1)
        except (Exception, KeyboardInterrupt):
            pass


def container_evaluator_result(container, prediction, testspec, timeout, grade, on_started, *, log_root=None):
    """Confirm launch before draining output; only inspected exits are terminal."""
    from runners.codex_exec import ProcessTerminal
    task_id = prediction["instance_id"]
    el._id(task_id)
    start = time.monotonic()
    terminal = ProcessTerminal(False, False, None)
    code, exec_id, start_issued, confirmation_attempted = None, None, False, False
    confirmation_failed, output = False, None

    def confirm(info):
        nonlocal confirmation_attempted, confirmation_failed
        require(info.get("Running") is True or
                (info.get("Running") is False and type(info.get("ExitCode")) is int),
                "ADAPTER_EVALUATOR_TERMINAL_UNKNOWN")
        if not confirmation_attempted:
            confirmation_attempted = True
            try:
                on_started()
            except BaseException:
                confirmation_failed = True
                _isolate_evaluator(container)
                raise

    try:
        ds.put_bytes(container, "/eval.sh", testspec.eval_script.encode())
        api = container.client.api
        created = api.exec_create(container.id, ["timeout", "--kill-after=10s", str(timeout),
                                                "/bin/bash", "/eval.sh"], workdir="/testbed")
        exec_id = created["Id"]
        start_issued = True
        output = api.exec_start(exec_id, stream=True, detach=False)
        info = api.exec_inspect(exec_id)
        confirm(info)
        # The synchronous response has acknowledged exec_start. Its output is
        # drained only AFTER the durable start-confirmed callback succeeded.
        with tempfile.TemporaryDirectory(prefix="wgm-grader-", dir=log_root) as temporary:
            # This pinned SWE-bench grader selects its repository parser from
            # the log's parent directory, which must be the exact task ID.
            log_directory = Path(temporary) / task_id
            log_directory.mkdir()
            path = log_directory / "test_output.txt"
            with path.open("wb") as stream:
                for chunk in ([output] if isinstance(output, bytes) else output):
                    require(type(chunk) is bytes, "ADAPTER_EVALUATOR_OUTPUT_INVALID")
                    stream.write(chunk)
            info = api.exec_inspect(exec_id)
            require(info.get("Running") is False and type(info.get("ExitCode")) is int,
                    "ADAPTER_EVALUATOR_TERMINAL_UNKNOWN")
            terminal = ProcessTerminal(True, True, info["ExitCode"], timed_out=info["ExitCode"] in (124, 137))
            if not terminal.timed_out:
                report = grade(testspec, prediction, path, include_tests_status=True)
                require(type(report) is dict and list(report) == [prediction["instance_id"]], "ADAPTER_GRADER_MISMATCH")
                resolved = report[prediction["instance_id"]]["resolved"]
                require(type(resolved) is bool, "ADAPTER_GRADER_MISMATCH")
                applied = report[prediction["instance_id"]].get("patch_successfully_applied")
                require(type(applied) is bool and (applied or not prediction["model_patch"]), "ADAPTER_GRADER_MISMATCH")
                return {"terminal":terminal, "duration_seconds":time.monotonic()-start,
                        "instance_id":next(iter(report)), "resolved":resolved, "error_code":None}
    except (Exception, KeyboardInterrupt) as error:
        code = "EVALUATOR_FAILED"
        if not start_issued:
            terminal = ProcessTerminal(False, False, None, launch_failed=True)
        elif not terminal.reaped:
            # Confirmation failure poisons the journal owner. Do not retry the
            # callback or invent a terminal from an object/usage record.
            if not confirmation_failed:
                try:
                    info = container.client.api.exec_inspect(exec_id)
                    if info.get("Running") is False and type(info.get("ExitCode")) is int:
                        confirm(info)
                        terminal = ProcessTerminal(True, True, info["ExitCode"],
                            timed_out=info["ExitCode"] in (124, 137), interrupted=isinstance(error, KeyboardInterrupt))
                except (Exception, KeyboardInterrupt):
                    pass
            if not terminal.reaped and not confirmation_failed:
                _isolate_evaluator(container)
    finally:
        if output is not None and callable(getattr(output, "close", None)):
            try:
                output.close()
            except (Exception, KeyboardInterrupt):
                pass
    return {"terminal":terminal, "duration_seconds":time.monotonic()-start,
            "instance_id":prediction["instance_id"], "resolved":False, "error_code":code}
