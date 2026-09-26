"""Operator-authorized retry of a search evaluation whose model call hit a connection failure.

A downstream model call that ends with a CLI connection error (Claude Code
result ``terminal_reason=api_error`` with no HTTP status) after it has already
used workspace tools is a known failure, not an unknown outcome. The validated
ledger binds it as failed and the search stops (BOUND_FAILURE): it is never
recorded as an unresolved task. After review, this explicit command appends an
``attempt_abandoned`` event (reason ``audited_connection_failure_retry``) with an
audit artifact. The failed binding, its call record and events stay in the
chain; the evaluation then runs once more as a fresh attempt (new container,
new work directory, new call id) under the same plan identity.
"""
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re

import evaluation_ledger as io
from evaluation_plan import check, durable_json

REASON = io.CONNECTION_RETRY_REASON
ERROR_LINE = re.compile(r"^claude_code result error: (.*)$", re.M)


def applies(snapshot):
    return any(e.kind == 'attempt_abandoned' and io._decode(e.data).get('reason') == REASON
               for e in snapshot.events)


def _connection_error(log_path):
    """The last CLI result error in the run log must be a connection-level api_error."""
    text = Path(log_path).read_text(errors='replace')
    lines = ERROR_LINE.findall(text)
    check(bool(lines), 'EVAL_RETRY_NO_CLI_ERROR')
    last = lines[-1]
    check('terminal_reason=api_error' in last and 'api_error_status=' not in last, 'EVAL_RETRY_NOT_CONNECTION')
    return {'log_path': str(log_path), 'log_sha256': hashlib.sha256(text.encode()).hexdigest(), 'cli_error': last}


def authorized_spec(ledger, plan, inspection):
    snap = inspection.snapshot
    check(snap is not None and not inspection.error_code and plan.identity.stage == 'search', 'EVAL_RETRY_LEDGER')
    check([a.attempt_id for a in snap.attempts] == [f'attempt-{i+1:06d}' for i in range(len(snap.attempts))]
          and all(a.authorization_artifacts for a in snap.attempts[:-1]), 'EVAL_RETRY_AUTHORIZATION')
    for index, attempt in enumerate(snap.attempts):
        if not attempt.authorization_artifacts:
            continue
        check(len(attempt.authorization_artifacts) == 1, 'EVAL_RETRY_AUTHORIZATION')
        row = io._read_json(ledger._folder(plan.identity.evaluation_id) / attempt.authorization_artifacts[0].path)
        body = row['authorization']
        check(row['sha256'] == io.digest(body) and body['schema_version'] == 1 and body['reason'] == REASON
              and body['run_id'] == ledger.run_id and body['run_fingerprint'] == plan.run_fingerprint
              and body['plan_digest'] == io.digest(asdict(plan))
              and body['evaluation_id'] == plan.identity.evaluation_id
              and body['attempt_id'] == attempt.attempt_id and body['new_attempt_id'] == f'attempt-{index+2:06d}',
              'EVAL_RETRY_AUTHORIZATION')
    latest = snap.attempts[-1]
    if latest.state == 'attempt_abandoned':
        check(latest.cleanup_status == 'success', 'EVAL_RETRY_CLEANUP_REQUIRED')
        return plan.spec(f'attempt-{len(snap.attempts)+1:06d}')
    return plan.spec(latest.attempt_id)


def authorize_evaluation_retry(directory, log_path, *, acknowledge=False):
    from evaluation_scheduler import ExecutionRun
    check(acknowledge, 'EVAL_RETRY_ACKNOWLEDGEMENT_REQUIRED')
    directory = io._checked_path(directory)
    manifest = io._read_json(directory / 'run_manifest.json')
    check(manifest['codex'].get('provider') == 'claude_code', 'EVAL_RETRY_PROVIDER_UNSUPPORTED')
    run = ExecutionRun(directory, manifest)
    with run.lock():
        failed = []
        for plan in run.plans.read():
            inspected = run.ledger.inspect(plan.identity.evaluation_id, plan.identity)
            check(not inspected.error_code, 'EVAL_RETRY_LEDGER')
            snap = inspected.snapshot
            if snap is None or snap.state != 'bound' or plan.identity.stage != 'search':
                continue
            binding = next(io._decode(e.data) for e in reversed(snap.events) if e.kind == 'bound')
            if binding['outcome'] == 'failed':
                failed.append((plan, inspected, binding))
        check(len(failed) == 1, 'EVAL_RETRY_REQUIRES_ONE_FAILED_BINDING')
        plan, inspected, binding = failed[0]
        snap = inspected.snapshot
        a = snap.attempts[-1]
        check(binding['failure_stage'] == 'model' and binding['error_code'] == 'MODEL_FAILED'
              and a.model_status == 'failed' and not a.evaluator_call_id and a.cleanup_status == 'success',
              'EVAL_RETRY_NOT_A_MODEL_FAILURE')
        terminal = next(io._decode(e.data) for e in reversed(snap.events)
                        if e.kind == 'model_terminal' and e.attempt_id == a.attempt_id)
        evidence = terminal.get('evidence', {})
        check(evidence.get('started') is True and evidence.get('reaped') is True
              and not evidence.get('timed_out') and not evidence.get('interrupted'), 'EVAL_RETRY_TERMINAL_NOT_TRUSTED')
        record_path = directory / 'codex_calls' / f'{a.call_id}.json'
        record = io._read_json(record_path)
        check(record.get('phase') == 'downstream' and record.get('provider') == 'claude_code'
              and record.get('status') == 'failed' and record.get('successful_calls') == 0
              and record.get('evaluation_id') == plan.identity.evaluation_id, 'EVAL_RETRY_CALL_RECORD_MISMATCH')
        body = {'schema_version': 1, 'reason': REASON, 'run_id': manifest['run_id'],
                'run_fingerprint': plan.run_fingerprint, 'plan_digest': io.digest(asdict(plan)),
                'evaluation_id': plan.identity.evaluation_id, 'task_id': plan.identity.task_id,
                'attempt_id': a.attempt_id, 'new_attempt_id': f'attempt-{len(snap.attempts)+1:06d}',
                'head': snap.events[-1].sha256, 'failed_binding': binding,
                'call_id': a.call_id, 'call_record_sha256': hashlib.sha256(record_path.read_bytes()).hexdigest(),
                'terminal_evidence': evidence, 'connection_error': _connection_error(log_path),
                'authorized_at_utc': datetime.now(timezone.utc).isoformat()}
        value = {'authorization': body, 'sha256': io.digest(body)}
        folder = run.ledger._folder(plan.identity.evaluation_id)
        path = folder / 'artifacts' / a.attempt_id / 'connection-retry.json'
        check(not path.exists(), 'EVAL_RETRY_ALREADY_AUTHORIZED')
        durable_json(path, value)
        reference = io.Artifact(str(path.relative_to(folder)), hashlib.sha256(path.read_bytes()).hexdigest(),
                                path.stat().st_size)
        with run.ledger.session(plan.identity) as session:
            check(session.snapshot() == snap, 'EVAL_RETRY_STATE_CHANGED')
            session._append('attempt_abandoned', a.attempt_id, {'artifacts': [asdict(reference)], 'reason': REASON})
        return {'status': 'authorized', 'evaluation_id': plan.identity.evaluation_id, 'task_id': plan.identity.task_id,
                'abandoned_attempt': a.attempt_id, 'next_attempt': body['new_attempt_id'],
                'authorization_sha256': value['sha256'], 'cli_error': body['connection_error']['cli_error']}
