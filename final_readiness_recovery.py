"""Explicit final-only retry of proven local receipt rejection before process start."""
from dataclasses import asdict
from pathlib import Path
import hashlib
import shlex

import evaluation_ledger as io
from evaluation_plan import check, durable_json

REASON = 'audited_local_readiness_no_start'
CODE = 'READINESS_RECEIPT_REJECTED'


def evidence(ledger, plan, snapshot, attempt):
    """No absence-of-process inference: require an affirmative no-start terminal."""
    from evaluation_adapter import EvaluationLifecycleAdapter
    check(plan.identity.stage == 'final_benchmark' and snapshot.identity == plan.identity
          and attempt.preparation_status == 'success' and attempt.model_status == 'failed'
          and not attempt.model_start_confirmed and not attempt.evaluator_call_id
          and not attempt.evaluation_start_confirmed and not attempt.result_artifacts, 'READINESS_RETRY_UNSAFE')
    events = [e for e in snapshot.events if e.attempt_id == attempt.attempt_id]
    check(not any(e.kind == 'bound' for e in events), 'READINESS_RETRY_UNSAFE')
    terminal = next(e for e in events if e.kind == 'model_terminal')
    data = io._decode(terminal.data)
    proof = data.get('evidence', {})
    io.validate_execution_evidence(proof)
    check(proof['started'] is False and proof['launch_failed'] is True and proof['reaped'] is False
          and proof['returncode'] is None and not proof['timed_out'] and not proof['interrupted']
          and not any(e.kind in ('model_start_confirmed', 'evaluation_started', 'evaluation_start_confirmed')
                      for e in events), 'READINESS_RETRY_UNSAFE')
    folder = ledger._folder(plan.identity.evaluation_id)
    artifacts = folder / 'artifacts' / attempt.attempt_id
    allowed = {'preparation.json', 'readiness-error.safe.json', 'readiness-authorization.json'}
    check(all(p.name in allowed for p in artifacts.iterdir()), 'READINESS_RETRY_MODEL_ARTIFACTS')
    # A prepared workspace must still match the sealed pre-model state. This also
    # rejects unjournaled model output, rather than treating missing markers as proof.
    adapter = EvaluationLifecycleAdapter(ledger, None)
    prepared = adapter._prepared(plan.spec(attempt.attempt_id))
    work = folder / 'work' / attempt.attempt_id
    from downstream_lifecycle import workspace_fingerprint
    check(work.is_dir() or attempt.cleanup_status == 'success', 'READINESS_RETRY_WORKSPACE_MISSING')
    if work.exists():
        check(workspace_fingerprint(work / 'testbed') == prepared.workspace_fingerprint,
              'READINESS_RETRY_WORKSPACE_CHANGED')
    usage_path = ledger.root / 'codex_calls' / (attempt.call_id + '.json')
    usage = io._read_json(usage_path) if usage_path.exists() else None
    check(usage is None or (usage.get('call_id') == attempt.call_id
          and usage.get('evaluation_id') == plan.identity.evaluation_id
          and usage.get('started_calls') == 0 and usage.get('successful_calls') == 0),
          'READINESS_RETRY_USAGE')
    if attempt.model_error.code == CODE:
        check(len(attempt.model_artifacts) == 1 and attempt.model_artifacts[0].path ==
              f'artifacts/{attempt.attempt_id}/readiness-error.safe.json', 'READINESS_RETRY_EVIDENCE')
        row = io._read_json(folder / attempt.model_artifacts[0].path)
        check(row['context'] == plan.spec(attempt.attempt_id).envelope()
              and row['physical_process_started'] is False
              and row['boundary'] in ('usage.before_runner', 'runner.before_execute')
              and row['rejection']['schema_version'] == 1
              and row['rejection']['code'] == 'NETWORK_READINESS_RECEIPT_INVALID'
              and row['rejection']['reason'] in ('EXPIRED', 'CLOCK', 'AUTHORITY', 'HMAC', 'CONFIGURATION', 'SCHEMA'),
              'READINESS_RETRY_EVIDENCE')
        rejection = io.digest(row)
    else:
        check(False, 'READINESS_RETRY_EVIDENCE')
    return {'terminal_digest': terminal.sha256, 'usage_digest': io.digest(usage),
            'preparation_digest': io.digest(asdict(prepared)), 'rejection_digest': rejection}


def grant(ledger, plan, snapshot, attempt):
    from preparation_authorization import _scope, _next
    check(len(attempt.authorization_artifacts) == 1, 'READINESS_RETRY_AUTHORIZATION')
    artifact = attempt.authorization_artifacts[0]
    check(artifact.path == f'artifacts/{attempt.attempt_id}/readiness-authorization.json',
          'READINESS_RETRY_AUTHORIZATION')
    row = io._read_json(ledger._folder(plan.identity.evaluation_id) / artifact.path)
    body = row['authorization']
    event = next(e for e in snapshot.events if e.kind == 'attempt_abandoned' and e.attempt_id == attempt.attempt_id)
    check(row['sha256'] == io.digest(body) and body['schema_version'] == 1 and body['reason'] == REASON
          and all(body[k] == v for k, v in _scope(ledger, plan).items())
          and body['failed_attempt_id'] == attempt.attempt_id
          and body['new_attempt_id'] == _next(attempt.attempt_id)
          and body['head'] == snapshot.events[event.sequence-2].sha256
          and body['evidence'] == evidence(ledger, plan, snapshot, attempt), 'READINESS_RETRY_AUTHORIZATION')
    return row


def execution_spec(ledger, plan, inspection, *, require_cleanup=True):
    """One consumed grant cannot authorize another failure or another attempt."""
    from evaluation_scheduler import SchedulingBlocked
    snap = inspection.snapshot
    grants = []
    for attempt in snap.attempts:
        if attempt.authorization_artifacts:
            grants.append(grant(ledger, plan, snap, attempt))
    latest = snap.attempts[-1]
    if grants:
        expected = [f'attempt-{i+1:06d}' for i in range(len(snap.attempts))]
        check([a.attempt_id for a in snap.attempts] == expected
              and all(a.authorization_artifacts for a in snap.attempts[:-1]), 'READINESS_RETRY_ATTEMPTS')
        if latest.state == 'attempt_abandoned':
            check(not require_cleanup or latest.cleanup_status == 'success', 'READINESS_RETRY_CLEANUP_REQUIRED')
            return plan.spec(grants[-1]['authorization']['new_attempt_id'])
        if (latest.preparation_status != 'failed' and
                (latest.model_status != 'failed' or latest.model_error.code != CODE)):
            return plan.spec(latest.attempt_id)
        raise SchedulingBlocked('READINESS_REAUDIT_REQUIRED', plan)
    if latest.model_error:
        if latest.model_error.code == CODE:
            evidence(ledger, plan, snap, latest)
            raise SchedulingBlocked('READINESS_AUTHORIZATION_REQUIRED', plan)
    return None  # Ordinary model failures keep their original reconciliation semantics.


def issue(run, evaluation_id, implementation):
    from preparation_authorization import verify_run, _scope, _next
    plans = verify_run(run)
    matches = [p for p in plans if p.identity.evaluation_id == evaluation_id]
    check(len(matches) == 1, 'READINESS_RETRY_EVALUATION')
    plan = matches[0]
    with run.ledger.session(plan.identity) as session:
        snap = session.snapshot()
        check(snap.attempts and not snap.binding_id, 'READINESS_RETRY_STATE')
        for prior in snap.attempts:
            if prior.authorization_artifacts:
                grant(run.ledger, plan, snap, prior)
        attempt = snap.attempts[-1]
        if attempt.state == 'attempt_abandoned':
            row = grant(run.ledger, plan, snap, attempt)
            check(row['authorization']['implementation'] == implementation, 'READINESS_RETRY_IMPLEMENTATION')
            return row
        check(attempt.state == 'model_terminal', 'READINESS_RETRY_STATE')
        proof = evidence(run.ledger, plan, snap, attempt)
        body = {'schema_version': 1, 'reason': REASON, **_scope(run.ledger, plan),
                'failed_attempt_id': attempt.attempt_id, 'new_attempt_id': _next(attempt.attempt_id),
                'head': snap.events[-1].sha256, 'evidence': proof, 'implementation': implementation}
        row = {'authorization': body, 'sha256': io.digest(body)}
        path = run.ledger._folder(evaluation_id) / 'artifacts' / attempt.attempt_id / 'readiness-authorization.json'
        if path.exists():
            check(io._read_json(path) == row, 'READINESS_RETRY_STATE_CHANGED')
        else:
            durable_json(path, row)
        reference = io.Artifact(str(path.relative_to(run.ledger._folder(evaluation_id))),
                                hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_size)
        session._append('attempt_abandoned', attempt.attempt_id, {'reason': REASON, 'artifacts': [asdict(reference)]})
        return row


def implementation(run):
    """The final's own commit, checkpoint and fingerprint; resume across commits is refused."""
    from preparation_recovery import implementation_identity
    from run_metadata import ROOT, repository_snapshot
    manifest = run.manifest
    if 'comparison_final' in manifest:
        checkpoint = manifest['comparison_final']['selection']['checkpoint']
    else:
        checkpoint = manifest['v2_derived']['checkpoint']
    identity = implementation_identity(ROOT, manifest['repository']['commit'], repository_snapshot(ROOT))
    return {**identity, 'checkpoint': checkpoint, 'final_fingerprint': manifest['reproducibility_fingerprint']}


def authorize_final(directory, checkpoint, evaluation_id):
    from preparation_authorization import final_run
    run = final_run(directory, checkpoint, evaluation_id=evaluation_id)
    with run.lock():
        identity = implementation(run)  # Dynamic checkpoint + complete scientific identity, before writes.
        check(identity['checkpoint'] == checkpoint and len(run.plans.read()) == len(run.manifest['tasks']['ordered_ids']),
              'READINESS_RETRY_SCOPE')
        row = issue(run, evaluation_id, identity)
    args = ['bash', 'run.sh', 'final-cleanup-readiness', '--output-dir', str(Path(directory).resolve()),
            '--checkpoint', str(checkpoint)]
    return {'token': row['sha256'], 'authorization': row['authorization'],
            'physical_model_calls': 0, 'physical_evaluator_calls': 0,
            'recommended_command': shlex.join(args)}


def cleanup(directory, checkpoint, *, client=None):
    """Only verified granted attempts are removed; prepared siblings are suspended."""
    from preparation_authorization import final_run, verify_run
    from dgm_recovery import owned_resource, container_identity
    run = final_run(directory, checkpoint)
    with run.lock():
        identity = implementation(run)
        check(identity['checkpoint'] == checkpoint, 'READINESS_RETRY_SCOPE')
        plans = verify_run(run)
        selected, authorized = [], 0
        for plan in plans:
            inspected = run.ledger.inspect(plan.identity.evaluation_id, plan.identity)
            snap = inspected.snapshot
            if not snap or not snap.attempts or snap.binding_id:
                continue
            a = snap.attempts[-1]
            if a.state == 'attempt_abandoned':
                row = grant(run.ledger, plan, snap, a)
                check(row['authorization']['implementation'] == identity, 'READINESS_RETRY_IMPLEMENTATION')
                authorized += 1
            elif a.state != 'preparation_terminal' or a.preparation_status != 'success':
                check(a.state == 'reserved', 'READINESS_CLEANUP_BLOCKED')
                continue
            if a.cleanup_status == 'success':
                continue
            check(not a.model_start_confirmed and not a.evaluator_call_id, 'READINESS_CLEANUP_BLOCKED')
            prepared, work = owned_resource(run.ledger, plan, a.attempt_id)
            from downstream_lifecycle import workspace_fingerprint
            check(workspace_fingerprint(work / 'testbed') == prepared.workspace_fingerprint,
                  'READINESS_RETRY_WORKSPACE_CHANGED')
            selected.append((plan, a, prepared, work))
        check(authorized > 0, 'READINESS_CLEANUP_AUTHORIZATION_REQUIRED')
        external = client is None
        if external:
            import docker
            client = docker.from_env(timeout=60)
        try:
            verified = []
            for plan, a, prepared, work in selected:
                from docker.errors import NotFound
                try:
                    container = client.containers.get(prepared.resource_id)
                    owned = container_identity(container, plan.spec(a.attempt_id), prepared)
                except NotFound:
                    row = io._read_json(work / 'resource-suspension.json')
                    owned = row['identity']
                    check(a.state == 'attempt_abandoned' and row['sha256'] == io.digest(owned)
                          and owned['id'] == prepared.resource_id
                          and owned['context'] == plan.spec(a.attempt_id).preparation_envelope()
                          and owned['name'] == 'sweb.eval.' + plan.identity.task_id + '.' + plan.identity.evaluation_id,
                          'READINESS_RESOURCE_OWNERSHIP')
                    container = None
                receipt = work / 'resource-suspension.json'
                if receipt.exists():
                    check(io._read_json(receipt) == {'schema_version': 1, 'identity': owned,
                          'sha256': io.digest(owned)}, 'READINESS_RESOURCE_CHANGED')
                verified.append((plan, a, work, container, owned))
            # No destructive call until every resource's complete ownership verifies.
            for plan, a, work, container, owned in verified:
                path = work / 'resource-suspension.json'
                row = {'schema_version': 1, 'identity': owned, 'sha256': io.digest(owned)}
                if path.exists():
                    check(io._read_json(path) == row, 'READINESS_RESOURCE_CHANGED')
                else:
                    durable_json(path, row)
                if container is not None and container.status == 'running':
                    container.stop(timeout=10)
                if a.state == 'attempt_abandoned':
                    if container is not None:
                        container.remove()
                    with run.ledger.session(plan.identity) as session:
                        session.finish_cleanup(a.attempt_id)
            return {'removed': sum(a.state == 'attempt_abandoned' for _, a, _, _, _ in verified),
                    'suspended': sum(a.state != 'attempt_abandoned' for _, a, _, _, _ in verified),
                    'physical_model_calls': 0, 'physical_evaluator_calls': 0}
        finally:
            if external:
                client.close()
