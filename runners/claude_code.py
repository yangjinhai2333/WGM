"""Claude Code CLI backend with the same lifecycle contract as ``codex exec``.

``ClaudeCodeRunner`` subclasses ``CodexExecRunner`` and replaces only argv
construction, the child environment and final-message extraction. Process
start/terminal evidence, timeouts, redaction and ledger callbacks are shared,
so validated search and evaluation code treat both backends identically.

Execution contract (see CLAUDE_BACKEND.md for the comparison with Codex):

* ``claude -p`` headless mode, prompt on stdin, ``--output-format stream-json``.
* ``--safe-mode`` and ``--setting-sources ""``: no CLAUDE.md, hooks, plugins,
  skills, MCP servers or user/project settings can alter an experiment. An
  evolved agent therefore cannot open a policy channel via CLAUDE.md.
* Built-in tools are an explicit allowlist (no web, no subagents).
* The Claude Code Bash sandbox is not relied upon: it silently degrades to
  unsandboxed execution when its dependencies are missing. Instead the whole
  CLI runs inside bubblewrap with a read-only host filesystem; only the
  workdir (write mode), a private /tmp and the Claude login state are writable.
  Confinement cannot be unavailable silently: a missing bwrap fails the start.
"""

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
from typing import Optional

from .codex_exec import (
    CodexConfigurationError, CodexExecConfig, CodexExecRunner, CodexOutputError,
    CodexStartError, _clean_environment, _execute, _reject_json_constant, _stderr_summary,
)


PROVIDER = "claude_code"
CLAUDE_EFFORTS = ("low", "medium", "high", "xhigh", "max")
DEFAULT_WRITE_TOOLS = ("Bash", "Read", "Edit", "Write", "Glob", "Grep", "TodoWrite")
DEFAULT_READ_TOOLS = ("Read", "Glob", "Grep")
TOOL_NAME = re.compile(r"[A-Z][A-Za-z]{1,40}")
# Session markers of an enclosing Claude Code process. Inheriting them makes a
# child behave as a nested session; provider/base-URL overrides would silently
# change which model or account serves the experiment.
_DROPPED_PREFIXES = ("CLAUDE_CODE_", "ANTHROPIC_")
_DROPPED_NAMES = {"CLAUDECODE", "CLAUDE_PID", "CLAUDE_EFFORT"}
# Container runtime sockets: connecting needs no write mount and grants root.
_RUNTIME_SOCKETS = ("/var/run/docker.sock", "/run/docker.sock", "/run/containerd/containerd.sock",
                    "/run/podman/podman.sock", "/var/run/crio/crio.sock")
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
# Transcripts and history of other Claude sessions (including an operator's
# interactive session) are private; headless calls never need them.
_PRIVATE_STATE = ("projects", "history.jsonl", "file-history", "sessions", "session-env",
                  "shell-snapshots", "paste-cache", "todos", "backups", "debug")
# Configuration that could alter later calls if a model rewrote it.
_PROTECTED_CONFIG = ("settings.json", "settings.local.json", "CLAUDE.md", "hooks",
                     "agents", "commands", "skills", "plugins", "output-styles")


def validate_claude_effort(value):
    if value is not None and value not in CLAUDE_EFFORTS:
        raise CodexConfigurationError("reasoning_effort must be low, medium, high, xhigh, max or null.")


@dataclass
class ClaudeCodeConfig(CodexExecConfig):
    """``codex`` config section selecting ``provider: claude_code``.

    Inherits the lifecycle fields; ``sandbox``/``approve_for_me`` keep their
    meaning (read-only diagnosis versus workspace-write solving). ``asdict``
    includes ``provider`` and the tool/confinement contract, so manifests and
    fingerprints of Claude runs never equal those of Codex runs.
    """
    executable: str = "claude"
    provider: str = PROVIDER
    confinement: str = "bwrap"
    bwrap_executable: str = "bwrap"
    write_tools: list[str] = field(default_factory=lambda: list(DEFAULT_WRITE_TOOLS))
    read_tools: list[str] = field(default_factory=lambda: list(DEFAULT_READ_TOOLS))
    # Host paths replaced by empty mounts. Relative entries resolve against this
    # checkout and the repository's main worktree (e.g. results, datasets).
    hidden_paths: list[str] = field(default_factory=list)
    # Account quota handling (see CLAUDE_BACKEND.md). Before a launch, wait for
    # the window reset once the last observed utilization reaches the threshold.
    # A 429 attempt that used only side-effect-free tools is treated as not
    # started: it is recorded, and the call is relaunched after the reset.
    quota_wait_threshold: float = 0.95
    rate_limit_relaunches: int = 3
    rate_limit_fallback_wait_seconds: float = 900.0
    # A connection-level API failure (no HTTP status) with only side-effect-free
    # tools is also not started: wait until the API is reachable again (at most
    # network_wait_seconds, probing network_probe_url), then relaunch. Without a
    # configured network_probe_url such failures are not relaunched.
    network_relaunches: int = 3
    network_wait_seconds: float = 3600.0
    network_probe_url: str | None = None

    def __post_init__(self, network_readiness, readiness_cancel_event):
        super().__post_init__(network_readiness, readiness_cancel_event)
        if self.provider != PROVIDER:
            raise CodexConfigurationError("ClaudeCodeConfig requires provider: claude_code.")
        if self.confinement not in ("bwrap", "none"):
            raise CodexConfigurationError("confinement must be bwrap or none.")
        for tools in (self.write_tools, self.read_tools):
            if not isinstance(tools, list) or not all(isinstance(t, str) and TOOL_NAME.fullmatch(t) for t in tools):
                raise CodexConfigurationError("Tool allowlists must be lists of built-in tool names.")
        if not isinstance(self.hidden_paths, list) or not all(isinstance(p, str) and p and "\x00" not in p
                                                              for p in self.hidden_paths):
            raise CodexConfigurationError("hidden_paths must be a list of path strings.")
        if any(t in ("Bash", "Edit", "Write", "NotebookEdit") for t in self.read_tools):
            raise CodexConfigurationError("read_tools must not contain writing or command tools.")
        validate_claude_effort(self.reasoning_effort)
        if (isinstance(self.quota_wait_threshold, bool) or not isinstance(self.quota_wait_threshold, (int, float))
                or not 0 < self.quota_wait_threshold <= 1):
            raise CodexConfigurationError("quota_wait_threshold must be in (0, 1].")
        if type(self.rate_limit_relaunches) is not int or not 0 <= self.rate_limit_relaunches <= 10:
            raise CodexConfigurationError("rate_limit_relaunches must be an integer in [0, 10].")
        if type(self.network_relaunches) is not int or not 0 <= self.network_relaunches <= 10:
            raise CodexConfigurationError("network_relaunches must be an integer in [0, 10].")
        if (isinstance(self.network_wait_seconds, bool) or not isinstance(self.network_wait_seconds, (int, float))
                or not 0 < self.network_wait_seconds <= 6 * 3600):
            raise CodexConfigurationError("network_wait_seconds must be in (0, 21600].")
        if self.network_probe_url is not None and (not isinstance(self.network_probe_url, str)
                                                   or not self.network_probe_url.startswith("https://")):
            raise CodexConfigurationError("network_probe_url must be null or an https URL.")
        if (isinstance(self.rate_limit_fallback_wait_seconds, bool)
                or not isinstance(self.rate_limit_fallback_wait_seconds, (int, float))
                or not 0 < self.rate_limit_fallback_wait_seconds <= 6 * 3600):
            raise CodexConfigurationError("rate_limit_fallback_wait_seconds must be in (0, 21600].")


# Tools that cannot change the workspace or the host. An attempt that used only
# these before a rate-limit rejection had no side effects.
SIDE_EFFECT_FREE_TOOLS = frozenset({"Read", "Glob", "Grep", "TodoWrite", "StructuredOutput"})
# Latest account quota observation of this process (all runners share one account).
_QUOTA = {"observation": None}
_QUOTA_LOCK = threading.Lock()


def _events(stdout):
    text = stdout.decode("utf-8", errors="replace") if isinstance(stdout, bytes) else stdout
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("{"):
            try:
                event = json.loads(line, parse_constant=_reject_json_constant)
            except (ValueError, RecursionError):
                continue
            if isinstance(event, dict):
                yield event


def quota_observation(stdout):
    """Last ``rate_limit_event`` window of a transcript: status, utilization, reset."""
    found = None
    for event in _events(stdout):
        info = event.get("rate_limit_info") if event.get("type") == "rate_limit_event" else None
        if not isinstance(info, dict):
            continue
        windows = info.get("unifiedWindows") if isinstance(info.get("unifiedWindows"), dict) else {}
        window = windows.get(info.get("rateLimitType")) or {}
        utilization = window.get("utilization", info.get("utilization"))
        resets = window.get("resetsAt", info.get("resetsAt"))
        found = {"status": info.get("status") if isinstance(info.get("status"), str) else None,
                 "window": info.get("rateLimitType") if isinstance(info.get("rateLimitType"), str) else None,
                 "utilization": float(utilization) if isinstance(utilization, (int, float)) and not isinstance(utilization, bool) else None,
                 "resets_at": float(resets) if isinstance(resets, (int, float)) and not isinstance(resets, bool) else None}
    return found


def rate_limit_rejection(returncode, timed_out, stdout):
    """Evidence that an attempt was refused by the account limit without side effects.

    Returns None unless the final result is a 429/limit rejection, the process
    was not timed out, and every tool the model used is side-effect free.
    """
    if timed_out or returncode == 0:
        return None
    result, tools, turns = None, [], None
    for event in _events(stdout):
        if event.get("type") == "result":
            result = event
        elif event.get("type") == "assistant" and isinstance(event.get("message"), dict):
            content = event["message"].get("content")
            for block in content if isinstance(content, list) else ():
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    tools.append(block.get("name") if isinstance(block.get("name"), str) else "?")
    quota = quota_observation(stdout)
    failed = result is not None and result.get("is_error") is not False
    limited = failed and (result.get("api_error_status") == 429 or (quota or {}).get("status") == "rejected")
    # Connection-level failure after the CLI's own retries: no HTTP status at all.
    network = (failed and not limited and result.get("terminal_reason") == "api_error"
               and result.get("api_error_status") is None)
    if not (limited or network) or any(tool not in SIDE_EFFECT_FREE_TOOLS for tool in tools):
        return None
    usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
    turns = result.get("num_turns") if type(result.get("num_turns")) is int else None
    return {"kind": "rate_limit" if limited else "network",
            "api_error_status": result.get("api_error_status"), "tools_used": sorted(set(tools)),
            "tool_calls": len(tools), "num_turns": turns, "quota": quota,
            "output_tokens": usage.get("output_tokens") if type(usage.get("output_tokens")) is int else None}


def _claude_environment():
    environment = {name: value for name, value in _clean_environment().items()
                   if name not in _DROPPED_NAMES and not name.startswith(_DROPPED_PREFIXES)}
    # A CLI self-update mid-run would change the recorded version under resume.
    environment["DISABLE_AUTOUPDATER"] = "1"
    environment["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
    return environment


def _mask_runtime_sockets():
    """bwrap cannot mount over a socket, so replace each socket's directory.

    The directory (normally /run) becomes a tmpfs; every other entry is mounted
    back read-only (symlinks recreated), keeping DNS and similar state visible.
    """
    sockets = {os.path.realpath(p) for p in _RUNTIME_SOCKETS if os.path.lexists(p)}
    runtime_names = {"docker", "docker.pid", "containerd", "podman", "crio"}
    args, masked = [], []
    for parent in sorted({os.path.dirname(p) for p in sockets}):
        if parent == "/" or not os.path.isdir(parent):
            continue
        if any(os.path.commonpath([parent, done]) == done for done in masked):
            continue  # Already absent: runtime subdirectories are never re-mounted.
        masked.append(parent)
        args.extend(["--tmpfs", parent])
        try:
            names = sorted(os.listdir(parent))
        except OSError:
            continue  # Unlistable: leave the whole directory empty.
        for name in names:
            path = os.path.join(parent, name)
            if path in sockets or name in runtime_names:
                continue
            if os.path.islink(path):
                args.extend(["--symlink", os.readlink(path), path])
            else:
                args.extend(["--ro-bind-try", path, path])
    return args


def _result_event(stdout: bytes):
    """Last well-formed ``result`` event of a stream-json transcript, or None."""
    text = stdout.decode("utf-8", errors="replace") if isinstance(stdout, bytes) else stdout
    found = None
    for line in text.split("\n"):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line, parse_constant=_reject_json_constant)
        except (ValueError, RecursionError):
            continue
        if isinstance(event, dict) and event.get("type") == "result":
            found = event
    return found


def final_message(stdout, structured):
    """The agent's final reply as the Codex backend would have written it.

    For a schema request this is the JSON text of ``structured_output``. A result
    marked as an error supplies no final message.
    """
    event = _result_event(stdout)
    if event is None or event.get("is_error") is not False:
        return None
    if structured:
        if "structured_output" not in event:
            return None
        return json.dumps(event["structured_output"], ensure_ascii=False, allow_nan=False)
    value = event.get("result")
    return value if isinstance(value, str) else None


class ClaudeCodeRunner(CodexExecRunner):
    """One blocking ``claude -p`` execution per call, no retry or preflight."""

    TEMP_PREFIX = "hgm-claude-"
    LOGIN_ARGS = ("auth", "status")
    VERSION_PATTERN = rb"[0-9]+\.[0-9]+\.[0-9]+ \(Claude Code\)"
    DISPLAY_NAME = "Claude Code"

    def __init__(self, config: Optional[ClaudeCodeConfig] = None, *, readiness_gate=None):
        config = config if config is not None else ClaudeCodeConfig()
        if not isinstance(config, ClaudeCodeConfig):
            raise CodexConfigurationError("ClaudeCodeRunner requires a ClaudeCodeConfig.")
        super().__init__(config, readiness_gate=readiness_gate)

    def _validate_effort(self, value):
        validate_claude_effort(value)

    def _note_quota(self, stdout):
        observation = quota_observation(stdout)
        if observation is not None:
            with _QUOTA_LOCK:
                _QUOTA["observation"] = {**observation, "observed_at": time.time()}

    def _wait_until(self, target, reason):
        """Sleep until ``target`` (epoch seconds); returns seconds waited."""
        start = time.monotonic()
        cancel = getattr(self.config, "readiness_cancel_event", None)
        remaining = target - time.time()
        if remaining > 0:
            print(f"Claude account quota: {reason}; waiting until "
                  f"{time.strftime('%Y-%m-%d %H:%M:%S %z', time.localtime(target))}", file=sys.stderr, flush=True)
        while remaining > 0:
            step = min(remaining, 30.0)
            if cancel is not None:
                if cancel.wait(step):
                    raise KeyboardInterrupt
            else:
                time.sleep(step)
            remaining = target - time.time()
        return time.monotonic() - start

    def _network_reachable(self):
        import urllib.error
        import urllib.request
        try:
            urllib.request.urlopen(self.config.network_probe_url, timeout=10)
            return True
        except urllib.error.HTTPError:
            return True  # Any HTTP response means the API endpoint is reachable.
        except Exception:
            return False

    def _logged_in(self):
        """`claude auth status` reports an active login (never exposes its output)."""
        try:
            done = subprocess.run([shutil.which(self.config.executable, path=self._environment().get("PATH"))
                                   or self.config.executable, *self.LOGIN_ARGS],
                                  capture_output=True, timeout=30, env=self._environment())
            return done.returncode == 0 and json.loads(done.stdout or b"{}").get("loggedIn") is True
        except (OSError, ValueError, subprocess.TimeoutExpired):
            return False

    def _wait_for_network(self, deadline):
        """Probe every 30 s until the API answers and the account is logged in, or ``deadline``.

        A connection-level failure can also be an expired login (the request is
        refused at once); waiting for an operator's re-login avoids spending the
        relaunches in seconds. Returns seconds waited.
        """
        start = time.monotonic()
        cancel = getattr(self.config, "readiness_cancel_event", None)
        print("Claude API connection failed with no side effects (not started); waiting for the network and a valid login",
              file=sys.stderr, flush=True)
        warned = False
        while not (self._network_reachable() and self._logged_in()) and time.time() < deadline:
            if not warned and self._network_reachable():
                print("Claude account is not logged in: run `claude auth login` (or /login); waiting",
                      file=sys.stderr, flush=True)
                warned = True
            step = min(30.0, max(deadline - time.time(), 0.0))
            if cancel is not None:
                if cancel.wait(step):
                    raise KeyboardInterrupt
            else:
                time.sleep(step)
        return time.monotonic() - start

    def _reset_target(self, quota):
        resets = (quota or {}).get("resets_at")
        now = time.time()
        if resets is not None and now < resets <= now + 6 * 3600:
            return resets + 60.0
        return now + float(self.config.rate_limit_fallback_wait_seconds)

    def _launch(self, args, workdir, environment, prompt, timeout, *, on_started=None, on_terminal=None,
                before_terminal=None):
        """One logical call: at most 1 + rate_limit_relaunches side-effect-free attempts.

        The ledger sees one start (the first process) and one terminal (the final
        process). A rejected attempt's terminal is withheld only when the call is
        relaunched; if waiting is interrupted, that terminal is delivered first.
        """
        state = self._call_state()
        state.rejections, state.waited = [], 0.0
        with _QUOTA_LOCK:
            observed = _QUOTA["observation"]
        if observed and observed.get("resets_at") and observed["resets_at"] > time.time() and (
                observed.get("status") == "rejected"
                or (observed.get("utilization") or 0) >= self.config.quota_wait_threshold):
            state.waited += self._wait_until(observed["resets_at"] + 60.0,
                f"last observed {observed.get('window')} utilization {observed.get('utilization')}")
        attempt_started = on_started
        while True:
            captured = {}
            kwargs = {}
            if attempt_started is not None:
                kwargs["on_started"] = attempt_started
            if on_terminal is not None:
                kwargs["on_terminal"] = lambda proof: captured.__setitem__("terminal", proof)
                kwargs["before_terminal"] = lambda code, out: captured.__setitem__("before", (code, out))
            try:
                returncode, timed_out, stdout, stderr = _execute(args, workdir, environment, prompt, timeout, **kwargs)
            except BaseException:
                if "terminal" in captured:
                    on_terminal(captured["terminal"])
                raise
            attempt_started = None  # Later processes continue the same logical call.
            self._note_quota(stdout)
            evidence = rate_limit_rejection(returncode, timed_out, stdout)
            limit = (self.config.rate_limit_relaunches if evidence is None or evidence["kind"] == "rate_limit"
                     else self.config.network_relaunches if self.config.network_probe_url else 0)
            if evidence is None or sum(r["kind"] == evidence["kind"] for r in state.rejections) >= limit:
                if on_terminal is not None and "terminal" in captured:
                    if before_terminal is not None and "before" in captured:
                        before_terminal(*captured["before"])
                    on_terminal(captured["terminal"])
                return returncode, timed_out, stdout, stderr
            target = (self._reset_target(evidence["quota"]) if evidence["kind"] == "rate_limit"
                      else time.time() + float(self.config.network_wait_seconds))
            state.rejections.append({**evidence, "rejected_at": time.time(), "retry_after_epoch": target})
            try:
                if evidence["kind"] == "rate_limit":
                    state.waited += self._wait_until(target, "request rejected with no side effects (not started)")
                else:
                    state.waited += self._wait_for_network(target)
            except BaseException:
                if on_terminal is not None and "terminal" in captured:
                    on_terminal(captured["terminal"])
                raise

    def _call_state(self):
        # Per thread: a runner may be shared by concurrent evaluation workers.
        local = self.__dict__.setdefault("_local", threading.local())
        if not hasattr(local, "rejections"):
            local.rejections, local.waited = [], 0.0
        return local

    def _take_rejections(self):
        state = self._call_state()
        rejections, waited = state.rejections, state.waited
        state.rejections, state.waited = [], 0.0
        return rejections, waited

    def _environment(self):
        return _claude_environment()

    def _build_args(self, executable, workdir, schema_path, last_path, extras):
        config = self.config
        writing = config.sandbox == "workspace-write" and config.approve_for_me
        if config.sandbox == "workspace-write" and not config.approve_for_me:
            # Headless mode has nobody to approve; Codex would equally stall.
            raise CodexConfigurationError("Claude workspace-write requires approve_for_me.")
        add_dirs = []
        claude = [executable, "-p", "--output-format", "stream-json", "--verbose",
                  "--safe-mode", "--setting-sources", "", "--strict-mcp-config"]
        if config.ephemeral:
            claude.append("--no-session-persistence")
        if config.model:
            claude.extend(["--model", config.model])
        if config.reasoning_effort is not None:
            claude.extend(["--effort", config.reasoning_effort])
        tools = config.write_tools if writing else config.read_tools
        claude.extend(["--tools", ",".join(tools)])
        # Inside the confinement every allowed tool is permitted; read-only
        # calls additionally have no writing tool and a read-only filesystem.
        claude.extend(["--permission-mode", "bypassPermissions" if writing else "dontAsk"])
        if schema_path is not None:
            claude.extend(["--json-schema", Path(schema_path).read_text(encoding="utf-8")])
        index = 0
        while index < len(extras):
            option = extras[index]
            if option == "--add-dir":
                add_dirs.append(extras[index + 1])
                claude.extend(["--add-dir", extras[index + 1]])
                index += 2
            else:
                # --skip-git-repo-check/--color are Codex-only presentation flags.
                index += 2 if option == "--color" else 1
        if config.confinement == "none":
            return claude
        return self._confined(workdir, add_dirs, writing) + ["--"] + claude

    def _hidden_paths(self):
        roots = [REPOSITORY_ROOT]
        try:
            common = subprocess.run(["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "--git-common-dir"],
                                    capture_output=True, text=True, timeout=10, env=_clean_environment())
            lines = common.stdout.splitlines() if common.returncode == 0 else []
            if len(lines) == 1:
                # Relative to the checkout for the main worktree, absolute otherwise.
                roots.append((REPOSITORY_ROOT / lines[0]).resolve().parent)
        except (OSError, subprocess.TimeoutExpired):
            pass
        hidden = []
        for entry in self.config.hidden_paths:
            candidates = [Path(entry)] if os.path.isabs(entry) else [root / entry for root in roots]
            hidden.extend(path.resolve() for path in candidates)
        return list(dict.fromkeys(hidden))

    def _confined(self, workdir, add_dirs, writing):
        environment = self._environment()
        bwrap = shutil.which(self.config.bwrap_executable, path=environment.get("PATH", os.defpath))
        if bwrap is None:
            raise CodexStartError("bwrap confinement is configured but bwrap was not found.")
        home = Path(environment.get("HOME") or os.path.expanduser("~")).resolve()
        config_dir = Path(os.environ.get("CLAUDE_CONFIG_DIR") or home / ".claude")
        executable = Path(shutil.which(self.config.executable, path=environment.get("PATH", os.defpath))
                          or self.config.executable).resolve()
        # Namespaces, not sessions: --unshare-pid plus --die-with-parent make a
        # timeout kill of bwrap terminate every descendant of the CLI.
        args = [os.path.abspath(bwrap), "--die-with-parent", "--unshare-pid",
                "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp"]
        args.extend(_mask_runtime_sockets())
        # $HOME holds other credentials (Codex, SSH, cloud, Git hosts). Replace it
        # and restore read-only only the top-level entries that PATH and the CLI
        # binary need (e.g. ~/.local, a Conda installation).
        if home != Path("/") and home.is_dir():
            args.extend(["--tmpfs", str(home)])
            needed = [Path(entry).resolve() for entry in environment.get("PATH", "").split(os.pathsep)
                      if os.path.isabs(entry)] + [executable]
            restored = sorted({home / path.relative_to(home).parts[0] for path in needed
                               if path.is_relative_to(home) and path != home})
            for path in restored:
                if path.exists() and path != config_dir:
                    args.extend(["--ro-bind", str(path), str(path)])
        for path in self._hidden_paths():
            if path.is_dir():
                args.extend(["--tmpfs", str(path)])
            elif path.exists():
                args.extend(["--ro-bind", "/dev/null", str(path)])
        # Login state must stay writable for OAuth refresh; its configuration
        # entries are re-mounted read-only so a model cannot plant hooks.
        for path in (config_dir, home / ".claude.json", config_dir / ".claude.json"):
            if path.exists():
                args.extend(["--bind", str(path), str(path)])
        for name in _PROTECTED_CONFIG:
            path = config_dir / name
            if path.exists():
                args.extend(["--ro-bind", str(path), str(path)])
        for name in _PRIVATE_STATE:
            path = config_dir / name
            if path.is_dir():
                args.extend(["--tmpfs", str(path)])
            elif path.exists():
                args.extend(["--bind", "/dev/null", str(path)])
        # Workdir binds come last, so a workdir inside a hidden tree stays visible.
        mode = "--bind" if writing else "--ro-bind"
        for directory in (workdir, *add_dirs):
            args.extend([mode, str(directory), str(directory)])
        args.extend(["--setenv", "TMPDIR", "/tmp", "--chdir", str(workdir)])
        return args

    def _stage_last_message(self, last_path, structured):
        def publish(returncode, stdout):
            if returncode != 0:
                return
            message = final_message(stdout, structured)
            if message is None:
                return
            data = message.encode("utf-8")
            fd = os.open(last_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
        return publish

    def _read_last_message(self, last_path, stdout, structured):
        try:
            return final_message(stdout, structured)
        except (OSError, UnicodeError):
            raise CodexOutputError("Could not read Claude final message.") from None

    def _event_error_summary(self, events, prompt):
        """Only CLI error fields, never tool output or assistant prose."""
        messages = []
        for event in events:
            if not isinstance(event, dict) or event.get("type") != "result":
                continue
            if event.get("is_error") is False:
                continue
            parts = [f"result.{event.get('subtype')}"]
            status = event.get("api_error_status")
            if status is not None:
                parts[0] += f" api_error_status={status}"
            reason = event.get("terminal_reason")
            if isinstance(reason, str):
                parts.append(f"terminal_reason={reason}")
            errors = event.get("errors")
            if isinstance(errors, list):
                parts.extend(str(item)[:512] for item in errors if isinstance(item, str))
            # One line per field: redaction drops a whole credential-bearing line,
            # which must not also discard the error category.
            messages.extend(parts)
        summary = _stderr_summary("\n".join(messages), prompt)
        if summary:  # Surface the CLI error category in the run log (the caller may not re-raise it).
            print("claude_code result error: " + summary.replace("\n", " | "), file=sys.stderr, flush=True)
        return summary
