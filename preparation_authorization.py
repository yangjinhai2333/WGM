"""Explicit, state-bound final preparation grants. No external callbacks."""
from dataclasses import asdict
from pathlib import Path
import shlex

import evaluation_ledger as io
from evaluation_plan import check, durable_json

REASON = 'audited_final_preparation_retry_v2'


def _folder(ledger, plan):
    return io._checked_path(ledger._folder(plan.identity.evaluation_id) / 'preparation_retry_authorizations')


def records(ledger, plan):
    folder = _folder(ledger, plan)
    if not folder.exists():
        return []
    rows = []
    for path in sorted(folder.iterdir()):
        if path.name.startswith('.tmp-'):
            continue
        row = io._read_json(path)
        check(path.name == row['authorization']['failed_attempt_id'] + '.json', 'PREPARATION_AUTHORIZATION_PATH')
        rows.append(row)
    return rows


def _scope(ledger, plan):
    manifest = io._read_json(ledger.root / 'run_manifest.json')
    final = manifest.get('comparison_final', {})
    derived = manifest.get('v2_derived')
    if derived is not None:
        check(manifest['run_type'] == 'wgm_v2' and manifest['experiment']['algorithm'] == 'wgm_v2'
              and derived['schema_version'] == 1 and derived['domain'] in ('primary', 'secondary')
              and derived['budget'] == len(manifest['tasks']['ordered_ids']) and derived['sampling'] == 'fixed_final60_once'
              and derived['selection']['candidate_id'] == plan.identity.candidate_id,
              'PREPARATION_AUTHORIZATION_WGM_FINAL_ONLY')
        final = derived
        selection = {'checkpoint': derived['checkpoint'],
                     'checkpoint_digest': derived['canonical_checkpoint_digest'],
                     'selection_digest': io.digest(derived)}
    else:
        check(final.get('algorithm') == 'hgm' and final.get('profile') == 'legacy_code',
              'PREPARATION_AUTHORIZATION_HGM_FINAL_ONLY')
        selection = final['selection']
    check(plan.identity.stage == 'final_benchmark' and selection['selection_digest'] == plan.selection_digest
          and manifest['run_id'] == plan.identity.run_id
          and manifest['reproducibility_fingerprint'] == plan.run_fingerprint,
          'PREPARATION_AUTHORIZATION_SCOPE')
    # Worker history and timing are runtime metadata, not the scientific manifest identity.
    return {'run_id': plan.identity.run_id, 'run_directory': str(ledger.root), 'manifest_identity': plan.run_fingerprint,
            'comparison_identity': io.digest(final), 'checkpoint': selection['checkpoint'],
            'checkpoint_digest': selection['checkpoint_digest'], 'evaluation_id': plan.identity.evaluation_id,
            'task_id': plan.identity.task_id, 'selection_digest': plan.selection_digest,
            'plan_identity_digest': io.digest(asdict(plan.identity)), 'plan_digest': io.digest(asdict(plan))}


def _head(snapshot):
    last = snapshot.events[-1]
    return {'sequence': last.sequence, 'event_digest': last.sha256}


def _next(attempt_id):
    prefix, number = attempt_id.rsplit('-', 1)
    check(prefix == 'attempt' and number.isdigit() and len(number) == 6, 'PREPARATION_AUTHORIZATION_ATTEMPT')
    return f'attempt-{int(number)+1:06d}'


def validate(ledger, plan, snapshot, row):
    """A pending grant requires the exact head; reservation is its one consumption."""
    from preparation_recovery import _failure
    body = row['authorization']
    check(row['sha256'] == io.digest(body) and body['schema_version'] == 2 and body['reason'] == REASON,
          'PREPARATION_AUTHORIZATION_DIGEST')
    check(all(body[k] == v for k,v in _scope(ledger, plan).items()), 'PREPARATION_AUTHORIZATION_SCOPE')
    check(body['new_attempt_id'] == _next(body['failed_attempt_id']), 'PREPARATION_AUTHORIZATION_ATTEMPT')
    check(_failure(snapshot, body['failed_attempt_id']) ==
          (body['failed_terminal_digest'], body['cleanup_digest']), 'PREPARATION_AUTHORIZATION_EVIDENCE')
    head = body['ledger_head']
    n = head['sequence']
    check(type(n) is int and 0 < n <= len(snapshot.events)
          and snapshot.events[n-1].sha256 == head['event_digest'], 'PREPARATION_AUTHORIZATION_STATE_CHANGED')
    prefix = snapshot.events[:n]
    check(prefix[-1].kind == 'cleanup_terminal' and prefix[-1].attempt_id == body['failed_attempt_id']
          and not any(e.kind.startswith(('model_', 'evaluation_')) for e in prefix),
          'PREPARATION_AUTHORIZATION_UNSAFE')
    if len(snapshot.events) == n:
        check(snapshot.attempts[-1].attempt_id == body['failed_attempt_id'],
              'PREPARATION_AUTHORIZATION_STATE_CHANGED')
        return 'pending'
    # No intervening event can be covered by an old grant. The next durable
    # reservation, already protected by the evaluation lease, consumes it once.
    event = snapshot.events[n]
    check(event.kind == 'reserved' and event.attempt_id == body['new_attempt_id'],
          'PREPARATION_AUTHORIZATION_STATE_CHANGED')
    return 'consumed'


def execution_spec(ledger, plan, inspection, rows):
    from evaluation_scheduler import SchedulingBlocked
    snapshot = inspection.snapshot
    check(snapshot is not None, 'PREPARATION_AUTHORIZATION_STATE_CHANGED')
    attempts = snapshot.attempts
    check([a.attempt_id for a in attempts] == [f'attempt-{i+1:06d}' for i in range(len(attempts))],
          'PREPARATION_AUTHORIZATION_ATTEMPT')
    grants = {}
    for row in rows:
        validate(ledger, plan, snapshot, row)
        body = row['authorization']
        check(body['failed_attempt_id'] not in grants, 'PREPARATION_AUTHORIZATION_DUPLICATE')
        grants[body['failed_attempt_id']] = body['new_attempt_id']
    for previous, attempt in zip(attempts, attempts[1:]):
        check(grants.get(previous.attempt_id) == attempt.attempt_id, 'PREPARATION_AUTHORIZATION_MISSING')
    latest = attempts[-1]
    if latest.preparation_status == 'failed':
        if latest.attempt_id not in grants:
            raise SchedulingBlocked('PREPARATION_REAUDIT_REQUIRED', plan)
        return plan.spec(grants[latest.attempt_id])
    return plan.spec(latest.attempt_id)


def verify_run(run):
    """Audit every persisted plan without executing or consuming any of them."""
    from evaluation_scheduler import read_binding, SchedulingBlocked
    run.verify()
    plans = run.plans.read()
    by_id = {p.identity.evaluation_id:p for p in plans}
    for snapshot in run.ledger.evaluations():
        check(snapshot.identity.evaluation_id in by_id
              and snapshot.identity == by_id[snapshot.identity.evaluation_id].identity,
              'PREPARATION_AUTHORIZATION_UNCLAIMED_EVALUATION')
    for plan in plans:
        inspection = run.ledger.inspect(plan.identity.evaluation_id, plan.identity)
        decision = io.classify(inspection, plan.identity).value
        check(not inspection.error_code and not inspection.legacy and not inspection.read_only,
              'PREPARATION_AUTHORIZATION_LEDGER')
        if decision.startswith('BLOCK_'):
            raise SchedulingBlocked(decision, plan)
        if decision == 'SKIP_BOUND':
            read_binding(run.ledger, plan)
        for row in records(run.ledger, plan):
            validate(run.ledger, plan, inspection.snapshot, row)
    return plans


def issue(run, evaluation_id, implementation):
    """Run lease held. Append one immutable grant, without reserving an attempt."""
    from preparation_recovery import _failure
    from evaluation_scheduler import resume_decision
    plans = verify_run(run)
    selected = [p for p in plans if p.identity.evaluation_id == evaluation_id]
    check(len(selected) == 1, 'PREPARATION_AUTHORIZATION_EVALUATION')
    plan = selected[0]
    inspection = run.ledger.inspect(evaluation_id, plan.identity)
    check(io.classify(inspection, plan.identity).value == 'RETRY_PREPARATION'
          and resume_decision(plan, inspection, {}) == 'RETRY_PREPARATION', 'PREPARATION_AUTHORIZATION_NOT_ELIGIBLE')
    snapshot = inspection.snapshot
    attempt_id = snapshot.attempts[-1].attempt_id
    terminal, cleanup = _failure(snapshot, attempt_id)
    check(not any(e.kind.startswith(('model_', 'evaluation_')) for e in snapshot.events),
          'PREPARATION_AUTHORIZATION_UNSAFE')
    # Verify prior authorizations too.
    rows = records(run.ledger, plan)
    if rows:
        from evaluation_scheduler import SchedulingBlocked
        try:
            execution_spec(run.ledger, plan, inspection, rows)
        except SchedulingBlocked as error:
            check(error.code == 'PREPARATION_REAUDIT_REQUIRED', error.code)
    else:
        check(len(snapshot.attempts) == 1 and attempt_id == 'attempt-000001', 'PREPARATION_AUTHORIZATION_MISSING')
    scope = _scope(run.ledger, plan)
    path = _folder(run.ledger, plan) / (attempt_id + '.json')
    if path.exists():
        row = io._read_json(path)
        check(validate(run.ledger, plan, snapshot, row) == 'pending'
              and row['authorization']['implementation'] == implementation, 'PREPARATION_AUTHORIZATION_CHANGED')
        return row
    body = {'schema_version': 2, 'reason': REASON, **scope, 'failed_attempt_id': attempt_id,
            'new_attempt_id': _next(attempt_id), 'failed_terminal_digest': terminal,
            'cleanup_digest': cleanup, 'ledger_head': _head(snapshot), 'implementation': implementation}
    row = {'authorization': body, 'sha256': io.digest(body)}
    validate(run.ledger, plan, snapshot, row)
    io._mkdir(path.parent)
    durable_json(path, row)
    return row


def use_token(run, token, implementation):
    """Return None for an unknown token; a token must belong to this exact run."""
    matches = [(p,row) for p in run.plans.read() for row in records(run.ledger,p) if row['sha256'] == token]
    if not matches:
        return None
    check(len(matches) == 1, 'PREPARATION_AUTHORIZATION_TOKEN')
    verify_run(run)
    plan, row = matches[0]
    check(row['authorization']['implementation'] == implementation, 'PREPARATION_AUTHORIZATION_IMPLEMENTATION')
    inspection = run.ledger.inspect(plan.identity.evaluation_id, plan.identity)
    status = validate(run.ledger, plan, inspection.snapshot, row)
    if status == 'consumed':
        from evaluation_scheduler import SchedulingBlocked
        raise SchedulingBlocked('PREPARATION_AUTHORIZATION_CONSUMED', plan)
    return execution_spec(run.ledger, plan, inspection, records(run.ledger,plan))


def command(directory, checkpoint, *, token=None, evaluation_id=None, readiness=False, role='primary'):
    args = ['bash', 'run.sh', 'final-eval' if token else 'final-authorize-preparation-retry',
            '--output-dir', str(Path(directory).resolve()), '--checkpoint', str(checkpoint)]
    if token:
        args += ['--evaluation-workers', '2', '--retry-preparation', token]
        if role == 'secondary':
            args += ['--roles', 'secondary-only']
        if readiness:
            import os
            args += ['--network-readiness', '--network-readiness-url',
                     os.environ.get('WGM_NETWORK_PROBE_URL', 'PROBE_URL'),
                     '--network-readiness-max-wait-seconds', '600']
    else:
        args += ['--evaluation-id', evaluation_id]
    return shlex.join(args)


def final_run(directory, checkpoint, *, evaluation_id=None, role='primary'):
    """Locate only an existing final domain. Never bootstrap or run BAI/search."""
    from final_adapter import profile
    from evaluation_scheduler import ExecutionRun
    manifest = io._read_json(Path(directory) / 'run_manifest.json')
    base = Path(directory) / 'derived' / f'checkpoint-{checkpoint:06d}'
    if profile(manifest) != 'wgm_v2':
        return ExecutionRun(base / 'final', io._read_json(base / 'final/run_manifest.json'))
    check(role in ('primary', 'secondary'), 'PREPARATION_AUTHORIZATION_ROLE')
    roles = ('primary', 'secondary') if evaluation_id else (role,)
    matches = []
    for name in roles:
        path = base / name / 'run_manifest.json'
        if path.exists():
            run = ExecutionRun(path.parent, io._read_json(path))
            if evaluation_id is None or any(p.identity.evaluation_id == evaluation_id for p in run.plans.read()):
                matches.append(run)
    check(len(matches) == 1, 'PREPARATION_AUTHORIZATION_EVALUATION')
    return matches[0]


def verified_final_report(directory, checkpoint, run):
    from final_adapter import legacy_report, wgm_report
    derived = run.manifest.get('v2_derived')
    return (wgm_report(directory, checkpoint, derived['domain']) if derived is not None
            else legacy_report(directory, checkpoint))


def authorize_final(directory, checkpoint, evaluation_id):
    """CLI boundary: audit saved evidence, then write only the scoped grant."""
    from preparation_recovery import implementation_identity
    from run_metadata import ROOT, repository_snapshot
    run = final_run(directory, checkpoint, evaluation_id=evaluation_id)
    wgm = 'v2_derived' in run.manifest
    with run.lock():
        report = verified_final_report(directory, checkpoint, run)
        check((report['algorithm'], report['profile']) in (('hgm', 'legacy_code'), ('wgm_v2', 'wgm_v2')),
              'PREPARATION_AUTHORIZATION_FINAL_ONLY')
        check(len(run.plans.read()) == report['requested'] == len(run.manifest['tasks']['ordered_ids']),
              'PREPARATION_AUTHORIZATION_FROZEN_FINAL_REQUIRED')
        identity = implementation_identity(ROOT, run.manifest['repository']['commit'], repository_snapshot(ROOT),
                                           **({'wgm_final': True} if wgm else {}))
        row = issue(run, evaluation_id, identity)
    return {'authorization': row['authorization'], 'token': row['sha256'],
            'model_calls': 0, 'evaluator_calls': 0,
            'recommended_command': command(directory, checkpoint, token=row['sha256'],
                                           role=run.manifest.get('v2_derived', {}).get('domain', 'primary'),
                                           readiness=run.manifest.get('network_readiness', {}).get('enabled', False))}
