"""Whitelist Codex event accounting; no prompt, tool output or auth persistence."""

from dataclasses import asdict, replace
import time
import uuid
import re
from network_readiness import await_runner, ReadinessAbort

PHASES = ("diagnosis", "self_improvement", "downstream", "final_evaluation")
TOKEN_FIELDS = ("input_tokens", "cached_input_tokens", "output_tokens")
COUNT_FIELDS = ("attempted_calls", "started_calls", "successful_calls", "failed_calls", "timed_out_calls")


def _model_name(value):
    return (isinstance(value, str) and value and len(value) < 128
            and all(c.isalnum() or c in "-._/:" for c in value))


def _is_claude_transcript(events):
    return any(isinstance(e, dict) and (e.get("type") == "result" and "session_id" in e
               or e.get("type") == "system" and e.get("subtype") == "init" and "claude_code_version" in e)
               for e in events)


def claude_event_facts(events):
    """Claude Code stream-json accounting from the final ``result`` event.

    Token fields keep the Codex meaning: ``input_tokens`` counts every prompt
    token (uncached, cache writes and cache reads), ``cached_input_tokens`` the
    cache reads among them. ``observed_model`` is the session model announced by
    the ``system/init`` event; ``modelUsage`` may also list auxiliary models.
    ``cost_usd`` is the CLI's list-price figure, not an account charge.
    """
    results = [e for e in events if isinstance(e, dict) and e.get("type") == "result"]
    models = {e["model"] for e in events if isinstance(e, dict) and e.get("type") == "system"
              and e.get("subtype") == "init" and _model_name(e.get("model"))}
    usage = {key: None for key in TOKEN_FIELDS}
    cost = turns = None
    if results:
        value = results[-1].get("num_turns")
        # Agent turns inside this single CLI invocation (no external re-prompting).
        turns = value if type(value) is int and value >= 0 else None
        raw = results[-1].get("usage")
        if isinstance(raw, dict):
            parts = [raw.get(k) for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")]
            if all(type(v) is int and v >= 0 for v in parts):
                usage["input_tokens"] = sum(parts)
                usage["cached_input_tokens"] = parts[2]
            value = raw.get("output_tokens")
            if type(value) is int and value >= 0:
                usage["output_tokens"] = value
        value = results[-1].get("total_cost_usd")
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
            cost = float(value)
    auxiliary = sorted(name for e in results if isinstance(e.get("modelUsage"), dict)
                       for name in e["modelUsage"] if _model_name(name))
    usage["token_usage_available"] = any(usage[k] is not None for k in TOKEN_FIELDS)
    usage["token_usage_complete"] = all(usage[k] is not None for k in TOKEN_FIELDS)
    return {**usage, "observed_model": next(iter(models)) if len(models) == 1 else None,
            "observed_models": sorted(models), "observed_reasoning_effort": None,
            "provider": "claude_code", "cost_usd": cost, "billed_models": auxiliary, "num_turns": turns}


def event_facts(events):
    """Only CLI lifecycle fields are authoritative, never agent-message text."""
    if _is_claude_transcript(events):
        return claude_event_facts(events)
    turns, models, efforts = [], set(), set()
    for event in events:
        if not isinstance(event, dict):
            continue
        kind = event.get("type")
        if kind == "turn.completed":
            turns.append(event["usage"] if isinstance(event.get("usage"), dict) else {})
        if kind in ("thread.started", "turn.started", "turn.completed"):
            value = event.get("model")
            if isinstance(value, str) and value and len(value) < 128 and all(c.isalnum() or c in "-._/:" for c in value):
                models.add(value)
            value = event.get("reasoning_effort")
            if value in ("minimal", "low", "medium", "high", "xhigh"):
                efforts.add(value)
    usage = {}
    for key in TOKEN_FIELDS:
        values = [t.get(key) for t in turns]
        usage[key] = sum(values) if values and all(type(v) is int and v >= 0 for v in values) else None
    usage["token_usage_available"] = any(usage[k] is not None for k in TOKEN_FIELDS)
    usage["token_usage_complete"] = all(usage[k] is not None for k in TOKEN_FIELDS)
    return {**usage, "observed_model": next(iter(models)) if len(models) == 1 else None,
            "observed_models": sorted(models),
            "observed_reasoning_effort": next(iter(efforts)) if len(efforts) == 1 else None}


def aggregate_usage(records):
    result = {}
    for phase in (*PHASES, "total"):
        selected = [r for r in records if phase == "total" or r["phase"] == phase]
        row = {key: sum(r.get(key, 0) for r in selected) for key in COUNT_FIELDS}
        row["duration_seconds"] = sum(r.get("duration_seconds", 0) for r in selected)
        for key in TOKEN_FIELDS:
            values = [r.get(key) for r in selected]
            row[key] = sum(values) if values and all(v is not None for v in values) else None
        row["token_usage_available"] = any(row[k] is not None for k in TOKEN_FIELDS)
        row["token_usage_complete"] = all(row[k] is not None for k in TOKEN_FIELDS)
        row["calls_with_usage"] = sum(bool(r.get("token_usage_available")) for r in selected)
        result[phase] = row
    return result


def invoke(runner, request, phase, *, recorder=None, records=None, schema=None, context_stats=None):
    """Account for one invocation, including launch failure and interruption.

    The Popen callback is the authoritative start boundary. Returned execution
    results also prove a start (useful for injected offline runners). An exception
    without that evidence never increments started_calls.
    """
    try:
        receipt = await_runner(runner, request.readiness_receipt, recorder=recorder,
                               reject_stale=request.call_id is not None or request.readiness_receipt is not None)
    except ReadinessAbort as error:
        # This boundary precedes runner.run/Popen, so a no-start terminal is proof,
        # unlike an absent callback after an arbitrary runner exception.
        if request.on_terminal is not None:
            from runners.codex_exec import ProcessTerminal
            request.on_terminal(ProcessTerminal(False, False, None, launch_failed=True))
            if error.code == 'NETWORK_READINESS_RECEIPT_INVALID':
                error.not_started_boundary = 'usage.before_runner'
        raise
    if receipt is not None:
        request = replace(request, readiness_receipt=receipt)
    if phase == "downstream":
        phase = getattr(recorder, "call_phase", phase)
    if phase not in PHASES:
        raise ValueError("Unknown Codex accounting phase")
    config = getattr(runner, "config", None)
    explicit_id = getattr(request, "call_id", None)
    if explicit_id is not None and (not isinstance(explicit_id, str) or not re.fullmatch(r"[0-9a-f]{64}", explicit_id)):
        raise ValueError("Invalid explicit call identity")
    record = dict(call_id=explicit_id if explicit_id is not None else uuid.uuid4().hex, phase=phase, attempted_calls=1,
                  started_calls=0, successful_calls=0, failed_calls=0, timed_out_calls=0,
                  duration_seconds=0.0, status="attempting", **event_facts([]),
                  configured_model=getattr(config, "model", None) or None,
                  configured_reasoning_effort=getattr(config, "reasoning_effort", None))
    if getattr(config, "provider", None) is not None:
        # Codex records keep their historical key set; other providers are explicit.
        record["provider"] = config.provider
    if context_stats is not None:
        record["diagnosis_context"] = context_stats
    if receipt is not None:
        record["network_readiness_receipt"] = asdict(receipt)
    if phase == "final_evaluation":
        record["instance_id"] = recorder.task_id
        record["final_selection_digest"] = recorder.selection_digest
    def persist():
        if recorder is not None:
            if explicit_id is None:
                recorder.record_call(record)
            else:
                try:
                    recorder.record_call(dict(record))
                except Exception:
                    # Accounting is not the execution journal. Never replace
                    # process evidence or retry a model because a sink failed.
                    record["usage_error_code"] = "USAGE_WRITE_FAILED"
    def started():
        record.update(started_calls=1, status="running")
        if explicit_id is not None and request.on_started is not None:
            request.on_started()
        persist()
    terminals = []
    def terminal(proof):
        terminals.append(proof)
        if request.on_terminal is not None:
            request.on_terminal(proof)
    start = time.monotonic()
    # Validated calls already have a durable ledger marker. Rebuilding accounting
    # here can consume the entire short receipt TTL before Popen. Persist after
    # the start callback (or the trusted no-start terminal), never on this gap.
    if explicit_id is None or receipt is None:
        persist()
    execution = None
    succeeded = False
    try:
        invocation = replace(request, on_started=started)
        if explicit_id is not None:
            invocation = replace(invocation, on_terminal=terminal)
        result = runner.run(invocation) if schema is None else runner.run_json(invocation, schema=schema)
        execution = getattr(result, "execution", result)
        succeeded = bool(getattr(execution, "ok", True))
        if explicit_id is not None:
            succeeded = succeeded and bool(terminals and terminals[-1].started and terminals[-1].reaped
                and type(terminals[-1].returncode) is int and terminals[-1].returncode == 0
                and not any((terminals[-1].launch_failed, terminals[-1].timed_out, terminals[-1].interrupted)))
        if not hasattr(execution, "returncode"):
            execution = None  # Minimal injected JSON runners may omit execution evidence.
        return result
    except BaseException as error:
        execution = getattr(error, "result", None)
        if isinstance(error, TimeoutError):
            record["timed_out_calls"] = 1
        record["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        raise
    finally:
        if execution is not None:
            record.update(started_calls=(1 if explicit_id is None else record["started_calls"]), timed_out_calls=int(execution.timed_out),
                          **event_facts(execution.events))
            if getattr(execution, "rate_limit_rejections", None):
                # Relaunched side-effect-free attempts; the call itself counts once.
                record["rate_limit_rejections"] = list(execution.rate_limit_rejections)
        if explicit_id is not None and terminals:
            record.update(started_calls=int(terminals[-1].started), timed_out_calls=int(terminals[-1].timed_out))
        record.update(successful_calls=int(succeeded), failed_calls=int(not succeeded),
                      duration_seconds=(execution.duration_seconds if execution is not None
                                        else time.monotonic() - start))
        if record["status"] != "interrupted":
            record["status"] = "success" if succeeded else "failed"
        record["model_source"] = ("cli_event" if record["observed_model"] else
                                  "project_config" if record["configured_model"] else "account_default_unresolved")
        if explicit_id is None:
            if records is not None:
                records.append(dict(record))
            persist()
        else:
            persist()
            if records is not None:
                records.append(dict(record))
