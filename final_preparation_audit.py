"""Read-only final audit: no locks, bootstrap, callbacks or reconciliation."""
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import evaluation_ledger as io
from evaluation_scheduler import read_binding, resume_decision
from run_metadata import file_hash


def audit(directory, checkpoint, *, role='primary'):
    from preparation_authorization import final_run, verified_final_report
    directory = Path(directory)
    run = final_run(directory, checkpoint, role=role)
    folder = run.directory
    report = verified_final_report(directory, checkpoint, run)
    plans = run.plans.read()
    rows = []
    decisions = Counter()
    for plan in plans:
        inspection = run.ledger.inspect(plan.identity.evaluation_id, plan.identity)
        decision = io.classify(inspection, plan.identity).value
        decisions[decision] += 1
        snapshot = inspection.snapshot
        attempts = []
        if snapshot:
            for attempt in snapshot.attempts:
                events = [e for e in snapshot.events if e.attempt_id == attempt.attempt_id]
                kinds = {e.kind for e in events}
                error_artifacts = []
                for artifact in attempt.preparation_artifacts:
                    if artifact.path.endswith('/preparation-error.safe.json'):
                        error_artifacts.append(io._read_json(run.ledger._folder(plan.identity.evaluation_id) / artifact.path))
                work = run.ledger._folder(plan.identity.evaluation_id) / 'work' / attempt.attempt_id
                attempts.append({**asdict(attempt),
                    'model_marker': 'model_started' in kinds, 'evaluator_marker': 'evaluation_started' in kinds,
                    'work_directory_exists': work.exists(), 'preparation_error_artifacts': error_artifacts,
                    'events': [{**asdict(e), 'data': io._decode(e.data)} for e in events]})
        bound = read_binding(run.ledger, plan) if decision == 'SKIP_BOUND' else None
        receipt = run.ledger._folder(plan.identity.evaluation_id) / 'preparation_retry.json'
        rows.append({'plan': asdict(plan), 'ledger_decision': decision,
                     'scheduler_decision': resume_decision(plan, inspection, {}),
                     'inspection_error': inspection.error_code, 'attempts': attempts,
                     'binding': asdict(bound) if bound else None,
                     'retry_authorization': io._read_json(receipt) if receipt.exists() else None})
    files = [directory / 'search_state.json', directory / 'checkpoints' / f'{checkpoint:06d}.json',
             folder / 'run_manifest.json']
    implementation = folder / 'recovery_implementation.json'
    return {'schema_version': 1, 'read_only': True, 'report': report,
            'file_digests': {str(p.relative_to(directory)): file_hash(p) for p in files if p.exists()},
            'plans_digest': io.digest([asdict(p) for p in plans]), 'plan_count': len(plans),
            'decisions': dict(decisions), 'evaluations': rows,
            'recovery_implementation': io._read_json(implementation) if implementation.exists() else None}


def compact_audit(directory, checkpoint, *, role='primary'):
    """Only action-requiring evaluations; no task inputs, proxy details or raw exceptions."""
    from preparation_authorization import records, validate, command, final_run
    from preparation_recovery import _failure
    full = audit(directory, checkpoint, role=role)
    run = final_run(directory, checkpoint, role=role)
    plans = {p.identity.evaluation_id:p for p in run.plans.read()}
    evaluations = []
    for row in full['evaluations']:
        decision = row['ledger_decision']
        if decision in ('SKIP_BOUND', 'START_NEW'):
            continue
        identity = row['plan']['identity']
        plan = plans[identity['evaluation_id']]
        inspection = run.ledger.inspect(identity['evaluation_id'], plan.identity)
        attempt = row['attempts'][-1] if row['attempts'] else {}
        grants = records(run.ledger, plan)
        status, token = 'none', None
        if grants:
            grant = grants[-1]
            status = validate(run.ledger, plan, inspection.snapshot, grant)
            token = grant['sha256']
        elif row['retry_authorization']:
            grant = row['retry_authorization']['authorization']
            status = 'pending' if attempt.get('attempt_id') == grant['failed_attempt_id'] else 'consumed'
            token = grant['failed_terminal_digest']
        eligible = False
        if decision == 'RETRY_PREPARATION' and full['report']['ambiguous'] == 0:
            try:
                _failure(inspection.snapshot, attempt['attempt_id'])
                eligible = not any(e.kind.startswith(('model_', 'evaluation_')) for e in inspection.snapshot.events)
            except (ValueError, KeyError):
                pass
        error = attempt.get('preparation_error_artifacts') or []
        terminals = [e['data'] for e in attempt.get('events', []) if e['kind'] == 'preparation_terminal']
        failure = (terminals[-1].get('error') or {}) if terminals else {}
        failure_stage = 'preparation' if failure else None
        if attempt.get('model_error'):
            failure, failure_stage = attempt['model_error'], 'model'
        recommended = None
        if eligible:
            recommended = command(directory, checkpoint, token=token if status == 'pending' else None,
                role=role,
                evaluation_id=identity['evaluation_id'], readiness=run.manifest.get('network_readiness', {}).get('enabled', False))
        local_proof = None
        if attempt.get('model_error') and not inspection.error_code:
            from final_readiness_recovery import evidence, grant, CODE
            if failure.get('code') == CODE:
                local_proof = evidence(run.ledger, plan, inspection.snapshot, inspection.snapshot.attempts[-1])
                eligible = True
                import shlex
                mode = 'final-authorize-readiness-retry'
                args = ['--evaluation-id', identity['evaluation_id']]
                row['scheduler_decision'] = 'READINESS_AUTHORIZATION_REQUIRED'
                if attempt.get('authorization_artifacts'):
                    authorization = grant(run.ledger, plan, inspection.snapshot, inspection.snapshot.attempts[-1])
                    status, token = 'authorized', authorization['sha256']
                    mode, args = 'final-cleanup-readiness', []
                    row['scheduler_decision'] = ('RETRY_AUTHORIZED' if attempt.get('cleanup_status') == 'success'
                                                 else 'READINESS_RETRY_CLEANUP_REQUIRED')
                recommended = shlex.join(['bash', 'run.sh', mode, '--output-dir', str(directory),
                                         '--checkpoint', str(checkpoint), *args])
        evaluations.append({'task_id': identity['task_id'], 'evaluation_id': identity['evaluation_id'],
            'attempt_id': attempt.get('attempt_id'), 'ledger_decision': decision,
            'scheduler_decision': row['scheduler_decision'], 'failure_stage': failure_stage,
            'failure_code': failure.get('code'), 'exception_type': error[-1]['exception_type'] if error else None,
            'model_started': bool(attempt.get('model_marker') or attempt.get('model_start_confirmed')),
            'evaluator_started': bool(attempt.get('evaluator_marker') or attempt.get('evaluation_start_confirmed')),
            'model_start_confirmed': bool(attempt.get('model_start_confirmed')),
            'evaluator_start_confirmed': bool(attempt.get('evaluation_start_confirmed')),
            'local_no_start_evidence': local_proof,
            'cleanup': attempt.get('cleanup_status'), 'ambiguous': 'AMBIGUOUS' in decision,
            'retry_eligible': eligible, 'authorization_status': status, 'authorization_token': token,
            'recommended_command': recommended})
    summary = {k:full['report'][k] for k in ('requested','completed','resolved','ambiguous',
        'final_physical_model_calls','final_physical_evaluator_calls','checkpoint_digest','selection_digest')}
    return {'schema_version': 1, 'read_only': True, 'checkpoint': checkpoint, 'summary': summary,
            'decisions': full['decisions'], 'evaluations': evaluations}
