"""Opt-in runtime readiness. No model calls, algorithm state or proxy management."""
from dataclasses import asdict, dataclass, field, replace
import argparse
import base64
import hashlib
import hmac
import http.client
import json
import logging
import math
import os
import re
import secrets
import socket
import ssl
import time
import threading
import weakref
from types import SimpleNamespace
from urllib.parse import unquote, urlsplit


class ReadinessConfigurationError(ValueError):
    pass


class ReadinessAbort(BaseException):
    """Controller stop, deliberately outside ordinary failure/retry handlers."""
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _url(value, *, proxy=False):
    try:
        if (type(value) is not str or not value or len(value) > 4096
                or any(c.isspace() or ord(c) < 32 for c in value) or "\\" in value):
            raise ValueError
        row = urlsplit(value)
        if row.scheme not in (("http",) if proxy else ("https",)) or not row.hostname:
            raise ValueError
        row.hostname.encode("idna")
        port = row.port if row.port is not None else (80 if proxy else 443)
        if not 1 <= port <= 65535:
            raise ValueError
        if proxy and (row.path not in ("", "/") or row.query or row.fragment):
            raise ValueError
        if not proxy and (row.username is not None or row.password is not None):
            raise ValueError
        return row, port
    except (ValueError, TypeError, UnicodeError):
        raise ReadinessConfigurationError("NETWORK_PROXY_URI_INVALID" if proxy else "NETWORK_PROBE_URL_INVALID") from None


@dataclass(frozen=True)
class ReadinessConfig:
    enabled: bool = False
    probe_url: str | None = field(default=None, repr=False)
    poll_seconds: float = 15
    connect_timeout_seconds: float = 3
    request_timeout_seconds: float = 10
    max_wait_seconds: float | None = None
    receipt_ttl_seconds: float = 5

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ReadinessConfigurationError("NETWORK_READINESS_CONFIG_INVALID")
        for name in ("poll_seconds", "connect_timeout_seconds", "request_timeout_seconds", "receipt_ttl_seconds", "max_wait_seconds"):
            value = getattr(self, name)
            if value is None and name == "max_wait_seconds":
                continue
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ReadinessConfigurationError("NETWORK_READINESS_CONFIG_INVALID")
        if self.enabled or self.probe_url is not None:
            _url(self.probe_url)

    @property
    def sha256(self):
        return digest(asdict(self))

    def safe(self):
        return {k: v for k, v in asdict(self).items() if k != "probe_url"} | {
            "schema_version": 1, "config_digest": self.sha256, "probe_configured": self.probe_url is not None}


@dataclass(frozen=True)
class Receipt:
    schema_version: int
    config_digest: str
    success_time: float
    monotonic_time: float
    valid_for_seconds: float
    category: str
    receipt_digest: str

    # Public evidence only. Authorization additionally requires the exact issued
    # object in a live, process-local authority; JSON/pickle copies confer none.


_PROCESS_EPOCH = object()


def _after_fork():
    global _PROCESS_EPOCH
    _PROCESS_EPOCH = object()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


class ReceiptAuthority:
    """Memory-only ownership, deliberately neither serializable nor persistent."""
    def __init__(self):
        self._key = secrets.token_bytes(32)
        self.audit_id = secrets.token_hex(8)  # Diagnostic only; never an authorization credential.
        self._pid, self._epoch = os.getpid(), _PROCESS_EPOCH
        self._issued = {}
        self._lock = threading.RLock()

    def __reduce__(self):
        raise TypeError("NETWORK_READINESS_AUTHORITY_PROCESS_LOCAL")

    def _current(self):
        return self._pid == os.getpid() and self._epoch is _PROCESS_EPOCH

    def _signature(self, body):
        return hmac.new(self._key, digest(body).encode("ascii"), hashlib.sha256).hexdigest()

    def issue(self, config, wall, monotonic):
        if not self._current():
            raise ReadinessAbort("NETWORK_READINESS_AUTHORITY_PROCESS_LOCAL")
        body = dict(schema_version=1, config_digest=config.sha256, success_time=wall,
                    monotonic_time=monotonic, valid_for_seconds=config.receipt_ttl_seconds,
                    category="HTTPS_RESPONSE")
        receipt = Receipt(**body, receipt_digest=self._signature(body))
        key = id(receipt)
        def discard(reference):
            if not self._current():
                return
            with self._lock:
                if self._issued.get(key) is reference:
                    self._issued.pop(key, None)
        with self._lock:
            self._issued[key] = weakref.ref(receipt, discard)
        return receipt

    def valid(self, receipt, config, monotonic):
        return self.rejection(receipt, config, monotonic) is None

    def rejection(self, receipt, config, monotonic):
        # Check process ownership before touching a possibly inherited lock.
        if not self._current() or type(receipt) is not Receipt:
            return 'AUTHORITY'
        with self._lock:
            reference = self._issued.get(id(receipt))
            if reference is None or reference() is not receipt:
                return 'AUTHORITY'
        if type(receipt.schema_version) is not int or receipt.schema_version != 1:
            return 'SCHEMA'
        body = asdict(receipt); body.pop("receipt_digest")
        if not (receipt.config_digest == config.sha256 and receipt.category == "HTTPS_RESPONSE"
                and receipt.valid_for_seconds == config.receipt_ttl_seconds
                and all(type(v) in (int, float) and math.isfinite(v) for v in
                        (receipt.success_time, receipt.monotonic_time, receipt.valid_for_seconds))):
            return 'CONFIGURATION'
        if not (type(receipt.receipt_digest) is str
                and re.fullmatch(r"[0-9a-f]{64}", receipt.receipt_digest) is not None
                and hmac.compare_digest(self._signature(body), receipt.receipt_digest)):
            return 'HMAC'
        age = monotonic - receipt.monotonic_time
        return None if 0 <= age < receipt.valid_for_seconds else 'EXPIRED' if age >= 0 else 'CLOCK'


class ProxyConnectError(OSError):
    pass


def _http_status(stream):
    # Bounded HTTP/1.x status line. Do not retain/log reason phrases or headers.
    line = stream.readline(65537)
    if len(line) > 65536 or not re.fullmatch(
            rb"HTTP/1\.[01] ([1-5][0-9]{2})(?: [\t\x20-\x7e\x80-\xff]*)?\r\n", line):
        raise http.client.BadStatusLine("NETWORK_HTTP_INVALID")
    return int(line[9:12])


class _ProbeConnection(http.client.HTTPSConnection):
    """HTTPSConnection TLS verification with bounded, status-only probe parsing."""
    def _tunnel(self):
        # Only _url-validated hosts and internally generated auth headers enter.
        host = self._tunnel_host.encode("idna").decode("ascii")
        authority = f"[{host}]:{self._tunnel_port}" if ":" in host else f"{host}:{self._tunnel_port}"
        request = f"CONNECT {authority} HTTP/1.1\r\nHost: {authority}\r\n"
        request += "".join(f"{key}: {value}\r\n" for key, value in self._tunnel_headers.items()) + "\r\n"
        self.sock.settimeout(self.stage_timeout())
        self.send(request.encode("ascii"))
        try:
            with self.sock.makefile("rb") as stream:
                self.sock.settimeout(self.stage_timeout())
                if not 200 <= _http_status(stream) <= 299:
                    raise ProxyConnectError("NETWORK_PROXY_CONNECT_FAILED")
                total = 0
                for _ in range(101):
                    self.sock.settimeout(self.stage_timeout())
                    line = stream.readline(65537)
                    total += len(line)
                    if total > 65536 or not line.endswith(b"\r\n"):
                        raise ProxyConnectError("NETWORK_PROXY_CONNECT_FAILED")
                    if line == b"\r\n":
                        break
                    if not re.fullmatch(rb"[!#$%&'*+.^_`|~0-9A-Za-z-]+:[\t\x20-\x7e\x80-\xff]*\r\n", line):
                        raise ProxyConnectError("NETWORK_PROXY_CONNECT_FAILED")
                else:
                    raise ProxyConnectError("NETWORK_PROXY_CONNECT_FAILED")
        except http.client.HTTPException:
            raise ProxyConnectError("NETWORK_PROXY_CONNECT_FAILED") from None
        # HTTPSConnection.connect now performs target TLS, using remaining time.
        self.sock.settimeout(self.stage_timeout())

    def getresponse(self):
        # Even an informational 100 proves target reachability. The general
        # HTTPResponse parser would wait for a final response after 100.
        with self.sock.makefile("rb") as stream:
            return SimpleNamespace(status=_http_status(stream))


def proxy_from_environment(getenv=os.getenv):
    # Deliberately read only four keys. NO_PROXY never bypasses this explicit proxy.
    for key in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        value = getenv(key)
        if value: return _url(value, proxy=True)
    raise ReadinessConfigurationError("NETWORK_PROXY_MISSING")


def probe(config, *, remaining=None, getenv=os.getenv, connect=None, connection=None, monotonic=time.monotonic):
    """TCP proxy reachability, then verified target TLS via HTTP CONNECT; no redirects/body."""
    connect = connect or socket.create_connection
    connection = connection or _ProbeConnection
    conn = None
    phase = "proxy"
    started = monotonic()
    def limit(seconds):
        if remaining is None:
            return seconds
        left = remaining - (monotonic() - started)
        if left <= 0:
            raise TimeoutError
        return min(seconds, left)
    try:
        proxy, port = proxy_from_environment(getenv)
        target, target_port = _url(config.probe_url)
        phase = "connect"
        with connect((proxy.hostname, port), timeout=limit(config.connect_timeout_seconds)):
            pass
        # HTTPSConnection connects to the proxy, tunnels first, then verifies TLS
        # against the target hostname. No environment discovery or direct fallback.
        conn = connection(proxy.hostname, port, timeout=limit(config.connect_timeout_seconds),
                          context=ssl.create_default_context())
        conn.stage_timeout = lambda: limit(config.request_timeout_seconds)
        headers = {}
        if proxy.username is not None:
            credentials = unquote(proxy.username) + ":" + unquote(proxy.password or "")
            headers["Proxy-Authorization"] = "Basic " + base64.b64encode(credentials.encode()).decode()
        conn.set_tunnel(target.hostname.encode("idna").decode("ascii"), target_port, headers=headers)
        phase = "https"
        conn.connect()
        conn.sock.settimeout(limit(config.request_timeout_seconds))
        path = (target.path or "/") + ("?" + target.query if target.query else "")
        conn.request("HEAD", path, headers={"Connection": "close"})
        conn.sock.settimeout(limit(config.request_timeout_seconds))
        response = conn.getresponse()
        return "HTTPS_RESPONSE" if type(response.status) is int and 100 <= response.status <= 599 else "NETWORK_HTTP_INVALID"
    except ReadinessConfigurationError as error:
        return str(error)
    except socket.gaierror:
        return "NETWORK_DNS_FAILED"
    except (socket.timeout, TimeoutError):
        return "NETWORK_PROBE_TIMEOUT"
    except ssl.SSLError:
        return "NETWORK_TLS_FAILED"
    except ProxyConnectError:
        return "NETWORK_PROXY_CONNECT_FAILED"
    except http.client.BadStatusLine:
        return "NETWORK_HTTP_INVALID"
    except (OSError, http.client.HTTPException, ValueError):
        return "NETWORK_CONNECT_FAILED" if phase == "connect" else "NETWORK_HTTPS_FAILED"
    finally:
        if conn is not None:
            try: conn.close()
            except Exception: pass


CATEGORIES = frozenset(("DISABLED", "HTTPS_RESPONSE", "NETWORK_PROXY_MISSING", "NETWORK_PROXY_URI_INVALID",
    "NETWORK_PROBE_URL_INVALID", "NETWORK_DNS_FAILED", "NETWORK_PROBE_TIMEOUT", "NETWORK_TLS_FAILED",
    "NETWORK_CONNECT_FAILED", "NETWORK_PROXY_CONNECT_FAILED", "NETWORK_HTTPS_FAILED", "NETWORK_HTTP_INVALID", "NETWORK_PROBE_FAILED",
    "NETWORK_READINESS_TIMEOUT", "NETWORK_READINESS_CANCELLED"))
COUNTERS = ("probe_attempts", "wait_cycles", "total_wait_seconds", "timeout_count", "cancel_count", "receipt_count")


class ReadinessGate:
    def __init__(self, config=None, *, check=None, monotonic=None, wall=None, waiter=None, log=None, cancel_event=None,
                 authority=None):
        self.config = config or ReadinessConfig()
        self.check = check or probe
        self.monotonic, self.wall = monotonic or time.monotonic, wall or time.time
        self.cancel_event = cancel_event
        self.waiter = waiter or (cancel_event.wait if cancel_event is not None else time.sleep)
        self.log = log or logging.getLogger(__name__).info
        self.last_summary = None
        self.authority = authority if authority is not None else ReceiptAuthority()

    def accepts(self, receipt):
        return self.authority.valid(receipt, self.config, self.monotonic())

    def wait(self, receipt=None, *, observe=None, reject_stale=False):
        if not self.config.enabled:
            return None
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise ReadinessAbort("NETWORK_READINESS_CANCELLED")
        if self.accepts(receipt):
            return receipt
        if reject_stale:
            error = ReadinessAbort("NETWORK_READINESS_RECEIPT_INVALID")
            now = self.monotonic()
            error.receipt_rejection = {'schema_version': 1, 'code': error.code,
                'reason': self.authority.rejection(receipt, self.config, now),
                'authority_id': self.authority.audit_id, 'authority_pid': self.authority._pid,
                'validation_pid': os.getpid(), 'validation_thread': threading.get_ident(),
                'checked_monotonic': now, 'config_digest': self.config.sha256,
                'receipt': asdict(receipt) if type(receipt) is Receipt else None}
            raise error
        start = self.monotonic(); last_log = None
        summary = {"schema_version": 1, "enabled": True, **dict.fromkeys(COUNTERS, 0),
                   "last_safe_category": "NETWORK_PROBE_FAILED", "configuration": self.config.safe()}
        try:
            while True:
                if self.cancel_event is not None and self.cancel_event.is_set():
                    raise KeyboardInterrupt
                elapsed = self.monotonic() - start
                remaining = None if self.config.max_wait_seconds is None else self.config.max_wait_seconds - elapsed
                if remaining is not None and remaining <= 0:
                    summary.update(timeout_count=1, last_safe_category="NETWORK_READINESS_TIMEOUT")
                    raise ReadinessAbort("NETWORK_READINESS_TIMEOUT")
                summary["probe_attempts"] += 1
                try:
                    category = self.check(self.config, remaining=remaining)
                except Exception:
                    category = "NETWORK_PROBE_FAILED"  # Never persist exception text.
                if self.cancel_event is not None and self.cancel_event.is_set():
                    raise KeyboardInterrupt
                if type(category) is not str or category not in CATEGORIES: category = "NETWORK_PROBE_FAILED"
                summary["last_safe_category"] = category
                elapsed = self.monotonic() - start
                if self.config.max_wait_seconds is not None and elapsed >= self.config.max_wait_seconds:
                    summary.update(timeout_count=1, last_safe_category="NETWORK_READINESS_TIMEOUT")
                    raise ReadinessAbort("NETWORK_READINESS_TIMEOUT")
                if category == "HTTPS_RESPONSE":
                    receipt = self.authority.issue(self.config, self.wall(), self.monotonic())
                    summary["receipt_count"] = 1
                    self.log(f"network_readiness category=HTTPS_RESPONSE attempt={summary['probe_attempts']} wait_seconds={elapsed:.3f}")
                    return receipt
                if last_log is None or elapsed - last_log >= 60:
                    self.log(f"network_readiness category={category} attempt={summary['probe_attempts']} wait_seconds={elapsed:.3f}")
                    last_log = elapsed
                interval = self.config.poll_seconds
                if self.config.max_wait_seconds is not None:
                    interval = min(interval, self.config.max_wait_seconds - elapsed)
                summary["wait_cycles"] += 1
                self.waiter(interval)
        except KeyboardInterrupt:
            summary.update(cancel_count=1, last_safe_category="NETWORK_READINESS_CANCELLED")
            raise ReadinessAbort("NETWORK_READINESS_CANCELLED") from None
        finally:
            summary["total_wait_seconds"] = max(0, self.monotonic() - start)
            self.last_summary = dict(summary)
            if observe is not None:
                try: observe(summary)
                except KeyboardInterrupt:
                    summary.update(cancel_count=1, last_safe_category="NETWORK_READINESS_CANCELLED")
                    self.last_summary = dict(summary)
                    raise ReadinessAbort("NETWORK_READINESS_CANCELLED") from None
                except Exception: self.log("network_readiness category=RUNTIME_METADATA_WRITE_FAILED")


def enabled(config):
    return getattr(config, "network_readiness", None) is not None and config.network_readiness.enabled


def await_runner(runner, receipt=None, *, recorder=None, reject_stale=False):
    config = getattr(runner, "config", None)
    if not enabled(config): return None
    cancel_event = getattr(config, "readiness_cancel_event", None)
    try:
        if cancel_event is not None and cancel_event.is_set():
            raise ReadinessAbort("NETWORK_READINESS_CANCELLED")
        gate = getattr(runner, "readiness_gate", None)
        if gate is None or gate.config != config.network_readiness:
            gate = ReadinessGate(config.network_readiness, cancel_event=cancel_event)
            # High-level wait, usage and runner must use the SAME authority.
            runner.readiness_gate = gate
        elif cancel_event is not None:
            gate.cancel_event = cancel_event
            if gate.waiter is time.sleep:
                gate.waiter = cancel_event.wait
        result = gate.wait(receipt, observe=getattr(recorder, "record_readiness", None), reject_stale=reject_stale)
        if cancel_event is not None and cancel_event.is_set():
            raise ReadinessAbort("NETWORK_READINESS_CANCELLED")
        return result
    except ReadinessAbort as error:
        if cancel_event is not None:
            from cancellation import request
            request(cancel_event, error)
        raise
    except KeyboardInterrupt:
        # Also cover interruption during receipt validation or summary writing,
        # outside the polling loop. Never let origin turn it into observation 0.
        if cancel_event is not None:
            cancel_event.set()
        raise ReadinessAbort("NETWORK_READINESS_CANCELLED") from None


def cancellable_operation(config, operation):
    """Stop already-queued origin work before it can sample a new task/parent."""
    if not enabled(config):
        return operation  # Preserve the exact original callable when disabled.
    event = config.readiness_cancel_event
    if event is None:
        raise ReadinessConfigurationError("NETWORK_READINESS_CANCELLATION_REQUIRED")
    def run():
        if event.is_set():
            raise ReadinessAbort("NETWORK_READINESS_CANCELLED")
        try:
            return operation()
        except ReadinessAbort:
            event.set()
            raise
    return run


def add_cli(parser):
    parser.add_argument("--network-readiness", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--network-readiness-url")
    parser.add_argument("--network-readiness-poll-seconds", type=float)
    parser.add_argument("--network-readiness-max-wait-seconds")


def configure_cli(config, args):
    values = {}
    for arg, key in (("network_readiness", "enabled"), ("network_readiness_url", "probe_url"),
                     ("network_readiness_poll_seconds", "poll_seconds")):
        value = getattr(args, arg, None)
        if value is not None: values[key] = value
    maximum = getattr(args, "network_readiness_max_wait_seconds", None)
    if maximum is not None:
        try: values["max_wait_seconds"] = None if maximum == "unbounded" else float(maximum)
        except ValueError: raise ReadinessConfigurationError("NETWORK_READINESS_CONFIG_INVALID") from None
    config.network_readiness = replace(config.network_readiness, **values)
    config.codex.network_readiness = config.network_readiness


class CompletionRecorder:
    """Origin opt-in budget hook. Completion does not include readiness cancellation."""
    def __init__(self, parent, completed):
        self.parent, self.completed = parent, completed

    def record_call(self, row):
        if self.parent is not None: self.parent.record_call(row)

    def record_readiness(self, row):
        if self.parent is not None: self.parent.record_readiness(row)

    def refresh(self):
        if self.parent is not None: return self.parent.refresh()

    def task_completed(self):
        self.completed()


def task_completed(recorder):
    # Only the explicit wrapper is authorized to change origin budget.
    if isinstance(recorder, CompletionRecorder): recorder.task_completed()


def safe_command(arguments):
    result = []; private_next = False
    for arg in arguments:
        if private_next:
            result.append("[readiness URL omitted]"); private_next = False
        elif arg == "--network-readiness-url":
            result.append(arg); private_next = True
        elif arg.startswith("--network-readiness-url="):
            result.append("--network-readiness-url=[omitted]")
        else:
            result.append(arg)
    return result


def safe_configuration(config):
    result = config.to_dict()
    result["network_readiness"] = config.network_readiness.safe()
    return result
