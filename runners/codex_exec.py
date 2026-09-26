"""Synchronous Codex CLI backend, independent of HGM and its API clients.

Workdirs must be absolute. Relative schema/add-dir paths are resolved against
the request workdir, never the Python process's current directory. This module
does not log prompts, environment values, or CLI output; returned output can
contain task data and should only be persisted deliberately by the caller.
"""

from dataclasses import InitVar, dataclass, field
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tempfile
import time
from typing import Any, Callable, Optional, Union
from network_readiness import ReadinessConfig, Receipt, await_runner, ReadinessAbort


PathLike = Union[str, Path]


def validate_reasoning_effort(value):
    # Fixed config key and official documented values; never accept arbitrary -c.
    if value is not None and value not in ("minimal", "low", "medium", "high", "xhigh"):
        raise CodexConfigurationError("reasoning_effort must be minimal, low, medium, high, xhigh or null.")


class CodexRunnerError(RuntimeError):
    """Base exception with a fixed, non-sensitive diagnostic."""


class CodexConfigurationError(CodexRunnerError, ValueError):
    """Invalid runner configuration or request."""


class CodexStartError(CodexRunnerError):
    """The executable could not be started or its pipes could not be used."""


class CodexOutputError(CodexRunnerError):
    """The final response is unavailable or is not valid JSON."""


@dataclass
class CodexExecConfig:
    executable: str = "codex"
    model: Optional[str] = None
    reasoning_effort: Optional[str] = None
    sandbox: str = "workspace-write"
    # Effective only with workspace-write: the CLI flag implies that sandbox
    # and conflicts with an explicit --sandbox (Codex CLI 0.154.0).
    approve_for_me: bool = True
    ephemeral: bool = True
    timeout_seconds: float = 3600.0
    json_output: bool = True
    output_schema: Optional[str] = None
    # Only --skip-git-repo-check, --color and --add-dir are accepted.
    # Arbitrary --config overrides could inject credentials or bypass sandboxing.
    extra_args: list[str] = field(default_factory=list)
    # Runtime-only InitVar: replace() preserves it; asdict()/scientific hashes exclude it.
    network_readiness: InitVar[Optional[ReadinessConfig]] = None
    readiness_cancel_event: InitVar[Any] = None

    def __post_init__(self, network_readiness, readiness_cancel_event):
        if network_readiness is not None and not isinstance(network_readiness, ReadinessConfig):
            from network_readiness import ReadinessConfigurationError
            raise ReadinessConfigurationError("NETWORK_READINESS_CONFIG_INVALID")
        if readiness_cancel_event is not None and not all(callable(getattr(readiness_cancel_event, name, None))
                                                          for name in ("set", "is_set", "wait")):
            from network_readiness import ReadinessConfigurationError
            raise ReadinessConfigurationError("NETWORK_READINESS_CONFIG_INVALID")
        self.network_readiness = network_readiness or ReadinessConfig()
        self.readiness_cancel_event = readiness_cancel_event


@dataclass(frozen=True)
class ProcessTerminal:
    """Positive local process evidence; never inferred from a missing PID."""
    started: bool
    reaped: bool
    returncode: Optional[int]
    timed_out: bool = False
    interrupted: bool = False
    launch_failed: bool = False


@dataclass(frozen=True)
class CodexExecRequest:
    prompt: str = field(repr=False)
    workdir: PathLike
    on_started: Optional[Callable[[], None]] = field(default=None, repr=False, compare=False)
    call_id: Optional[str] = None
    on_terminal: Optional[Callable[[ProcessTerminal], None]] = field(default=None, repr=False, compare=False)
    readiness_receipt: Optional[Receipt] = field(default=None, repr=False, compare=False)
    # Validated operation-owned quarantine. The terminal callback seals only
    # whitelisted fields and removes this raw staging file before publication.
    last_message_path: Optional[PathLike] = field(default=None, repr=False, compare=False)
    last_message_owner: Any = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class CodexExecResult:
    args: list[str]
    workdir: str
    returncode: int
    timed_out: bool
    stdout: str = field(repr=False)
    stderr: str = field(repr=False)
    last_message: Optional[str] = field(repr=False)
    events: list[Any] = field(repr=False)
    unparsed_lines: list[str] = field(repr=False)
    duration_seconds: float
    stderr_summary: str = field(default="", repr=False)
    event_error_summary: str = field(default="", repr=False)
    # Side-effect-free attempts rejected by an account rate limit before this
    # execution (Claude backend only); never used by the Codex backend.
    rate_limit_rejections: list = field(default_factory=list, repr=False)

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


class CodexExecutionError(CodexRunnerError):
    """run_json failed before parsing; the execution result remains available."""

    def __init__(self, result: CodexExecResult):
        self.result = result
        self.stderr_summary = result.stderr_summary
        self.event_error_summary = result.event_error_summary
        message = (
            "Codex execution timed out." if result.timed_out
            else f"Codex exited with status {result.returncode}."
        )
        if self.stderr_summary:
            message += f"\nstderr: {self.stderr_summary}"
        if self.event_error_summary:
            message += f"\nevents: {self.event_error_summary}"
        super().__init__(message)


@dataclass(frozen=True)
class CodexJSONResult:
    value: Any = field(repr=False)
    execution: CodexExecResult


@dataclass(frozen=True)
class CodexPreflightResult:
    executable: Optional[str]
    version_returncode: Optional[int] = None
    login_returncode: Optional[int] = None
    version_timed_out: bool = False
    login_timed_out: bool = False
    error: Optional[str] = None
    node_required: bool = False
    node_executable: Optional[str] = None
    node_returncode: Optional[int] = None
    node_timed_out: bool = False
    node_version: Optional[str] = None
    codex_version: Optional[str] = None

    @property
    def ok(self) -> bool:
        return (
            self.executable is not None and self.error is None
            and self.version_returncode == 0 and self.login_returncode == 0
            and not self.version_timed_out and not self.login_timed_out
            and (not self.node_required or (
                self.node_executable is not None and self.node_returncode == 0
                and not self.node_timed_out
            ))
        )


def _stderr_summary(stderr: str, prompt: str, *, limit: int = 3072) -> str:
    """Bounded diagnostics, redacted before truncation; never inspect auth files.

    CLI stderr is untrusted. Remove prompt echoes (including individual lines
    and JSON escapes), environment values and credential-bearing lines. Raw
    stderr remains available only in the result's repr-excluded field.
    """
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", stderr)
    private = {prompt, *prompt.splitlines()}
    # Include credentials filtered out of the child environment, without
    # storing this snapshot in a result or changing the parent's environment.
    for name, value in os.environ.items():
        if len(value) >= 4 or re.search(r"key|token|secret|password", name, re.I):
            private.add(value)
    # A blank diff/context line is formatting, not a secret. Replacing " "
    # would destroy every word boundary in an otherwise useful CLI error.
    private = {value for value in private if value.strip()}
    escaped = {json.dumps(value, ensure_ascii=True)[1:-1] for value in private}
    for value in sorted(private | escaped, key=len, reverse=True):
        text = text.replace(value, "[redacted]")
    sensitive_line = re.compile(
        r"api[ _-]?key|token|secret|password|authorization|cookie|credential|"
        r"auth\.json|sk-[\w-]+|eyJ[\w-]+\.[\w-]+\.[\w-]+", re.I,
    )
    lines = []
    for line in text.splitlines():
        if sensitive_line.search(line):
            line = "[credential-related stderr omitted]"
        lines.append("".join(char for char in line if char.isprintable() or char == "\t"))
    text = "\n".join(lines).strip()
    encoded = text.encode("utf-8")
    marker = "[truncated]\n"
    if len(encoded) > limit:
        text = marker + encoded[-(limit - len(marker)):].decode("utf-8", errors="ignore")
    return text


def _event_error_summary(events, prompt: str, *, limit: int = 3072) -> str:
    """Extract only CLI error messages, never tool output or agent prose."""
    messages = []
    for event in events:
        if not isinstance(event, dict):
            continue
        kind = event.get("type")
        if kind == "error":
            message = event.get("message")
        elif kind == "turn.failed" and isinstance(event.get("error"), dict):
            message = event["error"].get("message")
        else:
            continue
        if isinstance(message, str) and message.strip():
            messages.append(f"{kind}: {message}")
    return _stderr_summary("\n".join(messages), prompt, limit=limit)


def _clean_environment() -> dict[str, str]:
    """Keep account login support (HOME/CODEX_HOME etc.), drop API credentials."""
    provider_credentials = {
        "ANTHROPICAUTHTOKEN", "OPENAIAUTHTOKEN", "OPENROUTERAUTHTOKEN",
        "AZUREOPENAIADTOKEN", "AWSACCESSKEYID", "AWSSECRETACCESSKEY",
        "AWSSESSIONTOKEN", "AWSBEARERTOKENBEDROCK",
        "GOOGLEAPPLICATIONCREDENTIALS", "GOOGLEAUTHTOKEN", "AZUREACCESSTOKEN",
    }
    environment = {}
    for name, value in os.environ.items():
        normalized = name.upper().replace("_", "").replace("-", "")
        if ("APIKEY" not in normalized and normalized not in provider_credentials
                and not name.upper().startswith("AWS_")):
            environment[name] = value
    return environment


def _validate_timeout(timeout: float) -> None:
    if (
        isinstance(timeout, bool) or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout) or timeout <= 0
    ):
        raise CodexConfigurationError("Timeout must be a positive finite number.")


def _validate_text(value: str) -> None:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise CodexConfigurationError("CLI options must be nonempty text without NUL.")


def _resolve_path(path: PathLike, *, base: Optional[Path] = None) -> Path:
    try:
        resolved = Path(path).expanduser()
        if not resolved.is_absolute():
            if base is None:
                raise CodexConfigurationError("Workdir must be an absolute path.")
            resolved = base / resolved
        return resolved.resolve(strict=True)
    except (OSError, ValueError, TypeError, RuntimeError):
        raise CodexConfigurationError("Required path is invalid or does not exist.") from None


def _resolve_executable(executable: str, environment: dict[str, str]) -> str:
    _validate_text(executable)
    # Exclude relative PATH entries, including the empty entry for cwd.
    search_path = os.pathsep.join(
        entry for entry in environment.get("PATH", os.defpath).split(os.pathsep)
        if os.path.isabs(entry)
    )
    if not os.path.isabs(executable) and os.path.dirname(executable):
        raise CodexConfigurationError("Executable must be a name or absolute path.")
    found = shutil.which(executable, path=search_path)
    if found is None:
        raise CodexStartError("Codex executable not found or not executable.")
    return os.path.abspath(found)


def _extra_args(arguments: list[str], workdir: Path) -> list[str]:
    """Allow only options that cannot replace the managed execution contract."""
    if not isinstance(arguments, list):
        raise CodexConfigurationError("extra_args must be a list of CLI arguments.")
    result = []
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        _validate_text(argument)
        option, separator, value = argument.partition("=")
        if option == "--skip-git-repo-check" and not separator:
            result.append(option)
        elif option in {"--color", "--add-dir"}:
            if not separator:
                index += 1
                if index >= len(arguments):
                    raise CodexConfigurationError("An extra CLI option needs a value.")
                value = arguments[index]
            _validate_text(value)
            if option == "--color":
                if value not in {"always", "never", "auto"}:
                    raise CodexConfigurationError("Invalid color option.")
            else:
                directory = _resolve_path(value, base=workdir)
                if not directory.is_dir():
                    raise CodexConfigurationError("Additional workdir must be a directory.")
                value = str(directory)
            result.extend([option, value])
        else:
            # Includes --dangerously-bypass-approvals-and-sandbox, --yolo,
            # aliases, config/credential overrides, positional prompts and '--'.
            raise CodexConfigurationError("Unsupported or reserved extra CLI option.")
        index += 1
    return result


def _signal_group(process: subprocess.Popen, sig: int) -> None:
    try:
        if os.name == "posix":
            os.killpg(process.pid, sig)
        elif process.poll() is None:
            process.kill() if sig == signal.SIGKILL else process.terminate()
    except ProcessLookupError:
        pass


def _stop_and_collect(process: subprocess.Popen) -> tuple[bytes, bytes]:
    """Bound both graceful termination and draining pipes held by descendants."""
    _signal_group(process, signal.SIGTERM)
    try:
        try:
            return process.communicate(timeout=0.5)
        except subprocess.TimeoutExpired:
            _signal_group(process, signal.SIGKILL)
            try:
                return process.communicate(timeout=1.0)
            except subprocess.TimeoutExpired as error:
                # A descendant may have detached from the process group but kept
                # a pipe open. Do not let that defeat the caller's deadline.
                for stream in (process.stdin, process.stdout, process.stderr):
                    if stream is not None:
                        stream.close()
                process.wait(timeout=1.0)
                return error.output or b"", error.stderr or b""
    finally:
        # Also kill children which ignored TERM even if their parent exited and
        # closed its pipes before the grace period expired.
        _signal_group(process, signal.SIGKILL)


def _stop_and_reap(process):
    """A failed pipe drain must never skip the independent kill/wait attempt."""
    output = (b"", b"")
    try:
        output = _stop_and_collect(process)
    except BaseException:
        try:
            _signal_group(process, signal.SIGKILL)
        except BaseException:
            try:
                process.kill()
            except BaseException:
                pass
    try:
        process.wait(timeout=1.0)
    except BaseException:
        return output, False
    return output, type(process.returncode) is int


def _execute(args, workdir, environment, prompt, timeout, *, on_started=None, on_terminal=None,
             before_terminal=None):
    try:
        process = subprocess.Popen(
            args, cwd=str(workdir), env=environment,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=(os.name == "posix"),
        )
    except (OSError, ValueError):
        if on_terminal is not None:
            on_terminal(ProcessTerminal(False, False, None, launch_failed=True))
        raise CodexStartError("Could not start Codex executable.") from None
    if on_terminal is not None:
        # The owner retains the process until it has waited or exhausted a
        # bounded stop/reap attempt, including failures of either callback.
        try:
            if on_started is not None:
                on_started()
            stdout, stderr = process.communicate(input=prompt, timeout=timeout)
            process.wait(timeout=1.0)
        except BaseException as error:
            (stdout, stderr), reaped = _stop_and_reap(process)
            expired = isinstance(error, subprocess.TimeoutExpired)
            if reaped:
                on_terminal(ProcessTerminal(True, True, process.returncode, timed_out=expired,
                                            interrupted=isinstance(error, KeyboardInterrupt)))
                if expired:
                    return process.returncode, True, stdout, stderr
            raise
        # A failing terminal callback cannot start another process. The child
        # has already been reaped; only the ledger can commit that terminal.
        if type(process.returncode) is int:
            if before_terminal is not None:
                # Backends without a CLI last-message file publish it from
                # stdout here, so staged terminals read the same evidence.
                before_terminal(process.returncode, stdout)
            on_terminal(ProcessTerminal(True, True, process.returncode))
        return process.returncode, False, stdout, stderr

    # Ordinary calls retain their established behavior without ledger hooks.
    timed_out = False
    try:
        if on_started is not None:
            on_started()
        stdout, stderr = process.communicate(input=prompt, timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        stdout, stderr = _stop_and_collect(process)
    except BaseException:
        _stop_and_collect(process)
        raise
    return process.returncode, timed_out, stdout, stderr


def _reject_json_constant(value):
    raise ValueError("Non-JSON numeric constant.")


class CodexExecRunner:
    """One blocking CLI execution per call, with no automatic retry/preflight."""

    def __new__(cls, config: Optional[CodexExecConfig] = None, *args, **kwargs):
        # Existing call sites construct CodexExecRunner(config.codex). A config
        # selecting another provider yields that provider's runner subclass, so
        # every phase follows the one configured backend without call-site forks.
        if cls is CodexExecRunner and getattr(config, "provider", None) == "claude_code":
            from runners.claude_code import ClaudeCodeRunner
            cls = ClaudeCodeRunner
        return super().__new__(cls)

    def __init__(self, config: Optional[CodexExecConfig] = None, *, readiness_gate=None):
        self.config = config if config is not None else CodexExecConfig()
        self.readiness_gate = readiness_gate

    def run(self, request: CodexExecRequest) -> CodexExecResult:
        """Return stdout, stderr and status even on nonzero exit or timeout."""
        return self._run(request)

    def run_json(
        self, request: CodexExecRequest, *, schema: Optional[dict[str, Any]] = None
    ) -> CodexJSONResult:
        """Parse the final message as JSON; optionally supply an in-memory schema.

        Schema validation is delegated to Codex. Locally this checks JSON syntax,
        not JSON Schema conformance. A supplied schema file is never deleted.
        """
        execution = self._run(request, schema=schema)
        if not execution.ok:
            raise CodexExecutionError(execution)
        if execution.last_message is None:
            error = CodexOutputError("Codex did not produce a final message.")
            error.result = execution
            raise error
        try:
            value = json.loads(
                execution.last_message, parse_constant=_reject_json_constant
            )
        except (ValueError, RecursionError):
            error = CodexOutputError("Codex final message is not valid JSON.")
            error.result = execution
            raise error from None
        return CodexJSONResult(value=value, execution=execution)

    def _run(self, request, schema=None):
        config = self.config
        if request.call_id is not None and (not isinstance(request.call_id, str)
                                           or not re.fullmatch(r"[0-9a-f]{64}", request.call_id)):
            raise CodexConfigurationError("Invalid explicit call identity.")
        self._validate_effort(config.reasoning_effort)
        _validate_timeout(config.timeout_seconds)
        if config.sandbox not in ("read-only", "workspace-write"):
            raise CodexConfigurationError("Sandbox must be read-only or workspace-write.")
        for enabled in (config.approve_for_me, config.ephemeral, config.json_output):
            if not isinstance(enabled, bool):
                raise CodexConfigurationError("Runner switches must be booleans.")
        if config.model is not None and config.model != "":
            _validate_text(config.model)
            if config.model.startswith("-"):
                raise CodexConfigurationError("Model must be a name, not a CLI option.")
        if not isinstance(request.prompt, str):
            raise CodexConfigurationError("Prompt must be text.")
        try:
            prompt = request.prompt.encode("utf-8")
        except UnicodeError:
            raise CodexConfigurationError("Prompt must be valid UTF-8 text.") from None
        workdir = _resolve_path(request.workdir)
        if not workdir.is_dir():
            raise CodexConfigurationError("Workdir must be a directory.")
        extras = _extra_args(config.extra_args, workdir)
        schema_path = None
        if config.output_schema is not None:
            schema_path = _resolve_path(config.output_schema, base=workdir)
            if not schema_path.is_file():
                raise CodexConfigurationError("Output schema must be a file.")
        if schema is not None and schema_path is not None:
            raise CodexConfigurationError("Supply a schema file or inline schema, not both.")
        environment = self._environment()
        executable = _resolve_executable(config.executable, environment)
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix=self.TEMP_PREFIX) as temporary:
            last_path = Path(temporary) / "last-message.txt"
            staged_message = []
            terminal_callback = request.on_terminal
            if request.last_message_path is not None:
                from search_operation_journal import DiagnosisStaging
                last_path = Path(request.last_message_path)
                if (not last_path.is_absolute() or schema is None or request.call_id is None
                        or request.on_terminal is None or last_path.is_relative_to(workdir)
                        or type(request.last_message_owner) is not DiagnosisStaging):
                    raise CodexConfigurationError("Invalid durable output staging request.")
                owner = request.last_message_owner
                owner.start(last_path, request.call_id)
                def terminal_callback(proof):
                    if proof.started and proof.reaped and proof.returncode == 0 and not (
                            proof.timed_out or proof.interrupted or proof.launch_failed):
                        staged_message.append(owner.read().decode("utf-8", errors="replace"))
                    else:
                        staged_message.append(None)
                    request.on_terminal(proof)
            elif request.last_message_owner is not None:
                raise CodexConfigurationError("Invalid durable output staging request.")
            if schema is not None:
                if not isinstance(schema, dict):
                    raise CodexConfigurationError("Inline schema must be a JSON object.")
                try:
                    schema_text = json.dumps(schema, allow_nan=False)
                except (TypeError, ValueError, RecursionError):
                    raise CodexConfigurationError("Inline schema must be JSON serializable.") from None
                schema_path = Path(temporary) / "schema.json"
                schema_path.write_text(schema_text, encoding="utf-8")
            args = self._build_args(executable, workdir, schema_path, last_path, extras)
            before_terminal = self._stage_last_message(last_path, schema_path is not None)
            # High-level paths must already hold a fresh receipt before their ID/marker.
            # Never re-probe after such authorization; expiry fails closed instead.
            try:
                receipt = await_runner(self, request.readiness_receipt,
                                       reject_stale=request.call_id is not None or request.readiness_receipt is not None)
            except ReadinessAbort as error:
                if terminal_callback is not None:
                    terminal_callback(ProcessTerminal(False, False, None, launch_failed=True))
                    if error.code == 'NETWORK_READINESS_RECEIPT_INVALID':
                        error.not_started_boundary = 'runner.before_execute'
                raise
            if receipt is not None:
                started = time.monotonic()  # Readiness waiting is not model duration.
            returncode, timed_out, stdout, stderr = self._launch(
                args, workdir, environment, prompt, config.timeout_seconds,
                **({"on_started": request.on_started} if request.on_started is not None else {}),
                **({"on_terminal": terminal_callback} if terminal_callback is not None else {}),
                **({"before_terminal": before_terminal}
                   if before_terminal is not None and terminal_callback is not None else {}),
            )
            if staged_message:
                last_message = staged_message[-1]
            else:
                last_message = self._read_last_message(last_path, stdout, schema_path is not None)
        stdout = stdout.decode("utf-8", errors="replace")
        stderr = stderr.decode("utf-8", errors="replace")
        events, unparsed = [], []
        # JSONL uses LF delimiters. str.splitlines() also splits valid Unicode
        # characters (such as U+2028) embedded in JSON string values.
        lines = stdout.split("\n") if stdout else []
        if lines and lines[-1] == "":
            lines.pop()
        for line in lines:
            line = line.removesuffix("\r")
            if config.json_output:
                try:
                    events.append(json.loads(line, parse_constant=_reject_json_constant))
                    continue
                except (ValueError, RecursionError):
                    pass
            unparsed.append(line)
        rejections, waited = self._take_rejections()
        return CodexExecResult(
            args=args, workdir=str(workdir), returncode=returncode,
            timed_out=timed_out, stdout=stdout, stderr=stderr,
            last_message=last_message, events=events, unparsed_lines=unparsed,
            duration_seconds=time.monotonic() - started - waited,
            rate_limit_rejections=rejections,
            stderr_summary=_stderr_summary(stderr, request.prompt),
            event_error_summary=self._event_error_summary(events, request.prompt),
        )

    # Backend hooks. The Codex implementations below are the original contract;
    # alternative CLI backends override only argv construction and output parsing.
    TEMP_PREFIX = "hgm-codex-"
    LOGIN_ARGS = ("login", "status")
    VERSION_PATTERN = rb"codex-cli [0-9]+\.[0-9]+\.[0-9]+"
    DISPLAY_NAME = "Codex"

    def _validate_effort(self, value):
        validate_reasoning_effort(value)

    def _launch(self, *args, **kwargs):
        """Run one CLI process (Codex: exactly one, with no relaunch)."""
        return _execute(*args, **kwargs)

    def _take_rejections(self):
        """(rejected attempts, seconds spent waiting) for the finished execution."""
        return [], 0.0

    def _environment(self):
        return _clean_environment()

    def _build_args(self, executable, workdir, schema_path, last_path, extras):
        config = self.config
        args = [executable, "exec", "-C", str(workdir)]
        # --approve-for-me implies workspace-write and conflicts with
        # --sandbox even when the latter also says workspace-write.
        if config.approve_for_me and config.sandbox == "workspace-write":
            args.append("--approve-for-me")
        else:
            args.extend(["--sandbox", config.sandbox])
        if config.ephemeral:
            args.append("--ephemeral")
        if config.model:
            args.extend(["--model", config.model])
        if config.reasoning_effort is not None:
            args.extend(["-c", 'model_reasoning_effort=' + json.dumps(config.reasoning_effort)])
        if schema_path is not None:
            args.extend(["--output-schema", str(schema_path)])
        args.extend(["--output-last-message", str(last_path)])
        if config.json_output:
            args.append("--json")
        args.extend(extras)
        args.append("-")
        return args

    def _stage_last_message(self, last_path, structured):
        """Return a (returncode, stdout) hook run before a terminal callback, or None.

        Codex writes --output-last-message itself, so it needs no hook.
        """
        return None

    def _read_last_message(self, last_path, stdout, structured):
        try:
            return last_path.read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:
            return None
        except OSError:
            raise CodexOutputError("Could not read Codex final message.") from None

    def _event_error_summary(self, events, prompt):
        return _event_error_summary(events, prompt)

    def preflight(self, *, timeout_seconds: float = 10.0) -> CodexPreflightResult:
        """Read-only executable/Node/version/login checks; no model execution.

        Raw login output is deliberately not returned: it can identify accounts.
        Node is required only for a Node-shebang CLI (such as the npm launcher),
        not native executables or other launchers. Python is unrelated to this.
        Each command has its own timeout. A successful status does not test model
        availability, quota, or compatibility of every exec option.
        """
        _validate_timeout(timeout_seconds)
        environment = self._environment()
        try:
            executable = _resolve_executable(self.config.executable, environment)
        except (CodexConfigurationError, CodexStartError) as error:
            return CodexPreflightResult(executable=None, error=str(error))
        results = {"executable": executable}
        try:
            with open(executable, "rb") as stream:
                shebang = stream.readline(256)
        except OSError:
            shebang = b""
        results["node_required"] = bool(re.match(rb"#![^\n]*\bnode(?:\s|$)", shebang))
        commands = []
        if results["node_required"]:
            try:
                node = _resolve_executable("node", environment)
            except (CodexConfigurationError, CodexStartError):
                return CodexPreflightResult(**results, error="Node executable not found on PATH.")
            results["node_executable"] = node
            commands.append(("node", [node, "--version"]))
        commands.extend((("version", [executable, "--version"]),
                         ("login", [executable, *self.LOGIN_ARGS])))
        for name, command in commands:
            try:
                returncode, timed_out, stdout, _ = _execute(
                    command, Path(os.path.abspath(os.sep)),
                    environment, b"", timeout_seconds,
                )
            except CodexStartError:
                return CodexPreflightResult(**results, error=f"Could not start {name} check.")
            results[f"{name}_returncode"] = returncode
            results[f"{name}_timed_out"] = timed_out
            if timed_out or returncode != 0:
                detail = "timed out" if timed_out else f"exited with status {returncode}"
                return CodexPreflightResult(**results, error=f"{self.DISPLAY_NAME} {name} check {detail}.")
            if name in ("node", "version"):
                pattern = rb"v[0-9]+\.[0-9]+\.[0-9]+" if name == "node" else self.VERSION_PATTERN
                version = re.search(rb"(?m)^" + pattern + rb"$", stdout)
                if version:
                    results["node_version" if name == "node" else "codex_version"] = version[0].decode("ascii")
        return CodexPreflightResult(**results)
