"""Explicit DGM interruption audit/waiver; no model or evaluator entry point.

An operator waiver abandons UNKNOWN evidence, never fabricates a failed result.
Only marker-without-confirmation can be waived. Confirmed starts stay blocked.
"""
from dataclasses import asdict
from collections import Counter
from pathlib import Path
import hashlib
import re

import evaluation_ledger as io
from evaluation_plan import check, durable_json
from evaluation_scheduler import ExecutionRun, read_binding
from evaluation_adapter import EvaluationLifecycleAdapter, validate_preparation_context

REASON = 'operator_accepts_unknown_model_outcome'
IMPLEMENTATION_REASON = 'dgm_cancellation_evidence_and_explicit_operator_waiver_v1'


def implementation(manifest):
    """Recovery runs on the exact, clean commit that produced the run."""
    from run_metadata import ROOT, repository_snapshot
    repo = repository_snapshot(ROOT)
    check(repo['tracked_clean'] and not repo['untracked_paths'], 'DGM_RECOVERY_DIRTY')
    check(repo['commit'] == manifest['repository']['commit'], 'DGM_RECOVERY_COMMIT')
    return {'old_commit': manifest['repository']['commit'], 'new_commit': repo['commit'],
            'reason': IMPLEMENTATION_REASON}


def record_implementation(run, row):
    path = run.directory / 'dgm_recovery_implementation.json'
    if path.exists():
        check(io._read_json(path) == row, 'DGM_RECOVERY_IMPLEMENTATION_CHANGED')
    else:
        durable_json(path, row)


def validate_manifest(manifest):
    """Use build_manifest's schema, including its duplicated scientific identity."""
    check(type(manifest.get('schema_version')) is int and manifest['schema_version'] == 1,
          'DGM_RECOVERY_MANIFEST_SCHEMA')
    experiment = manifest.get('experiment', {})
    check(manifest.get('run_type') == 'dgm' and experiment.get('algorithm') == 'dgm'
          and experiment.get('dgm_profile') == 'legacy_code', 'DGM_RECOVERY_PROFILE')
    configuration = manifest.get('configuration', {})
    effective = configuration.get('effective', {})
    check(effective.get('search') == experiment
          and configuration.get('sha256') == io.digest(effective), 'DGM_RECOVERY_CONFIGURATION')


def open_run(directory):
    from dgm_legacy_search import read_checkpoint, report
    from search_operation_journal import SearchOperationJournal
    from candidate_provenance import read_chain
    directory = io._checked_path(directory)
    manifest = io._read_json(directory / 'run_manifest.json')
    validate_manifest(manifest)
    run = ExecutionRun(directory, manifest)
    checkpoint = read_checkpoint(directory, manifest['reproducibility_fingerprint'])
    # DGM checkpoint v1 has no top-level profile. It is bound by fingerprint;
    # its summary and frozen N2 plans must also agree with the manifest.
    state = checkpoint['state']
    check(state.get('summary', {}).get('algorithm') == 'dgm'
          and 0 < state['target_budget'] <= manifest['experiment']['parameters']['planning_horizon'],
          'DGM_RECOVERY_CHECKPOINT')
    def validate_operation(plan):
        identity = plan.identity
        check(identity['algorithm'] == 'dgm' and identity['profile'] == 'legacy_code'
              and identity['run_id'] == manifest['run_id']
              and identity['run_fingerprint'] == manifest['reproducibility_fingerprint'],
              'DGM_RECOVERY_OPERATION_IDENTITY')
    from search_operation_journal import SearchOperationPlan
    for generation in state['generations']:
        for child in generation['children']:
            validate_operation(SearchOperationPlan.from_dict(child['operation_plan']))
    report(directory, manifest)  # Validates archive, consumed bindings and N2 terminals.
    journal = SearchOperationJournal(directory, manifest['run_id'], manifest['reproducibility_fingerprint'])
    for plan in journal.plans():
        validate_operation(plan)
        state, _ = journal.inspect(plan)
        if state['terminal'] and state['terminal']['outcome'] == 'candidate':
            read_chain(directory, plan.identity['candidate_id'], require_versioned=True)
    return run, checkpoint


def audit(directory, *, compact=False):
    run, checkpoint = open_run(directory)
    rows = []
    for plan in run.plans.read():
        inspected = run.ledger.inspect(plan.identity.evaluation_id, plan.identity)
        check(not inspected.error_code and not inspected.read_only, 'DGM_RECOVERY_LEDGER')
        snap = inspected.snapshot
        a = snap.attempts[-1] if snap and snap.attempts else None
        binding = None
        if snap and snap.state == 'bound':
            read_binding(run.ledger, plan)
            binding = next(io._decode(e.data) for e in reversed(snap.events) if e.kind == 'bound')
            # The strict reader above validates the waiver, execution identity,
            # terminal and artifacts. Business evidence comes from this binding.
            a = next(attempt for attempt in snap.attempts if attempt.attempt_id == binding['attempt_id'])
        patch = next((asdict(ref) for ref in a.model_artifacts
                      if ref.path == f'artifacts/{a.attempt_id}/patch.diff'), None) if binding else None
        rows.append({'evaluation_id': plan.identity.evaluation_id, 'task_id': plan.identity.task_id,
            'candidate_id': plan.identity.candidate_id, 'decision': io.classify(inspected).value,
            'plan_digest': io.digest(asdict(plan)), 'plan_identity_digest': io.digest(asdict(plan.identity)),
            'binding_digest': snap.binding_id if binding else None,
            'binding_identity_digest': binding['identity_sha256'] if binding else None,
            'binding_outcome': binding['outcome'] if binding else None,
            'bound_attempt_id': binding['attempt_id'] if binding else None,
            'patch': patch, 'attempt_count': len(snap.attempts) if snap else 0,
            'head': snap.events[-1].sha256 if snap else None,
            'attempt_id': a.attempt_id if a else None, 'model_call_id': a.call_id if a else None,
            'model_started': a.model_start_confirmed if a else False,
            'model_terminal': a.model_status if a else None,
            'model_terminal_digest': a.model_terminal_digest if a else None,
            'evaluator_started': a.evaluation_start_confirmed if a else False,
            'evaluator_marker': a.evaluator_call_id if a else None,
            'evaluator_terminal': a.evaluation_status if a else None,
            'evaluator_terminal_digest': a.evaluation_terminal_digest if a else None,
            'cleanup': a.cleanup_status if a else None})
    result = {'checkpoint_digest': checkpoint['sha256'], 'rng_digest': io.digest(checkpoint['state']['rng']),
            'manifest_fingerprint': run.manifest['reproducibility_fingerprint'],
            'valid_evaluations': checkpoint['state']['valid_evaluations'], 'evaluations': rows,
            'ambiguous': sum('AMBIGUOUS' in r['decision'] for r in rows)}
    if compact:
        # Planned but unstarted tasks at a budget boundary are counted, not errors.
        result['decision_counts'] = dict(Counter(r['decision'] for r in rows))
        result['evaluations'] = [r for r in rows if r['attempt_count'] > 1
                                or r['decision'] not in ('SKIP_BOUND', 'START_NEW')]
        result['target_budget'] = checkpoint['state']['target_budget']
        result['archive'] = checkpoint['state']['archive']
    return result


def owned_resource(ledger, plan, attempt_id):
    spec = plan.spec(attempt_id)
    adapter = EvaluationLifecycleAdapter(ledger, None)
    prepared = adapter._prepared(spec)  # Artifact, execution identity and digest.
    work = adapter._workdir(spec)
    resource = io._read_json(work / 'resource.json')
    validate_preparation_context(spec, resource['context'])
    check(resource['resource_id'] == prepared.resource_id, 'DGM_RESOURCE_IDENTITY')
    return prepared, work


def container_identity(container, spec, prepared):
    container.reload()
    expected_name = 'sweb.eval.' + spec.identity.task_id + '.' + spec.identity.evaluation_id
    check(re.fullmatch(r'[0-9a-f]{64}', prepared.resource_id) is not None
          and container.id == prepared.resource_id and container.name == expected_name,
          'DGM_RESOURCE_OWNERSHIP')
    return {'id': container.id, 'name': container.name, 'image': container.attrs['Image'],
            'context': spec.preparation_envelope()}


def authorized_spec(ledger, plan, inspection, *, require_cleanup=True):
    manifest = io._read_json(ledger.root / 'run_manifest.json')
    validate_manifest(manifest)
    check(plan.identity.stage == 'search'
          and manifest['reproducibility_fingerprint'] == plan.run_fingerprint, 'DGM_RECOVERY_IDENTITY')
    snap = inspection.snapshot
    check(snap is not None and not inspection.error_code, 'DGM_RECOVERY_LEDGER')
    check([a.attempt_id for a in snap.attempts] == [f'attempt-{i+1:06d}' for i in range(len(snap.attempts))]
          and all(a.authorization_artifacts for a in snap.attempts[:-1]), 'DGM_RECOVERY_AUTHORIZATION')
    for index, attempt in enumerate(snap.attempts):
        if not attempt.authorization_artifacts:
            continue
        check(len(attempt.authorization_artifacts) == 1, 'DGM_RECOVERY_AUTHORIZATION')
        reference = attempt.authorization_artifacts[0]
        row = io._read_json(ledger._folder(plan.identity.evaluation_id) / reference.path)
        body = row['authorization']
        event = next(e for e in snap.events if e.kind == 'attempt_abandoned' and e.attempt_id == attempt.attempt_id)
        check(row['sha256'] == io.digest(body) and body['schema_version'] == 1
            and body['reason'] == REASON and body['run_directory'] == str(ledger.root)
            and body['run_fingerprint'] == plan.run_fingerprint and body['run_id'] == ledger.run_id
            and body['plan_digest'] == io.digest(asdict(plan))
            and body['evaluation_id'] == plan.identity.evaluation_id and body['task_id'] == plan.identity.task_id
            and body['attempt_id'] == attempt.attempt_id and body['head'] == snap.events[event.sequence-2].sha256
            and body['new_attempt_id'] == f'attempt-{index+2:06d}', 'DGM_RECOVERY_AUTHORIZATION')
        usage_path = ledger.root / 'codex_calls' / (attempt.call_id + '.json')
        usage = io._read_json(usage_path) if usage_path.exists() else None
        check(io.digest(usage) == body['usage_digest'], 'DGM_RECOVERY_USAGE_CHANGED')
    latest = snap.attempts[-1]
    if latest.state == 'attempt_abandoned':
        check(not require_cleanup or latest.cleanup_status == 'success', 'DGM_RECOVERY_CLEANUP_REQUIRED')
        return plan.spec(f'attempt-{len(snap.attempts)+1:06d}')
    return plan.spec(latest.attempt_id)


def authorize(directory, evaluation_id, *, acknowledge=False):
    check(acknowledge, 'DGM_UNKNOWN_OUTCOME_ACKNOWLEDGEMENT_REQUIRED')
    run, checkpoint = open_run(directory)
    with run.lock():
        from dgm_legacy_search import read_checkpoint
        checkpoint = read_checkpoint(run.directory, run.manifest['reproducibility_fingerprint'])
        implementation_row = implementation(run.manifest)
        from search_operation_journal import SearchOperationJournal
        journal = SearchOperationJournal(run.directory, run.manifest['run_id'],
                                         run.manifest['reproducibility_fingerprint'])
        for operation in journal.plans():
            state, _ = journal.inspect(operation)
            check(state['blocked'] is None and all('marker' not in a or 'terminal' in a
                  for stage in ('diagnosis', 'generation') for a in state[stage]),
                  'DGM_RECOVERY_OPERATION_BLOCKED')
        plans = run.plans.read()
        plan = next((p for p in plans if p.identity.evaluation_id == evaluation_id), None)
        check(plan is not None, 'DGM_RECOVERY_EVALUATION')
        for p in plans:
            inspected = run.ledger.inspect(p.identity.evaluation_id, p.identity)
            decision = io.classify(inspected).value
            check(not inspected.error_code and (p == plan or not decision.startswith('BLOCK_')),
                  'DGM_RECOVERY_OTHER_BLOCKED')
        inspected = run.ledger.inspect(evaluation_id, plan.identity)
        snap = inspected.snapshot
        check(snap is not None and snap.attempts, 'DGM_RECOVERY_STATE')
        a = snap.attempts[-1]
        if a.state == 'attempt_abandoned':
            authorized_spec(run.ledger, plan, inspected, require_cleanup=False)
            record_implementation(run, implementation_row)
            return {'status': 'already_authorized', 'evaluation_id': evaluation_id}
        check(io.classify(inspected).value == 'BLOCK_AMBIGUOUS_MODEL_ATTEMPT'
              and not a.model_start_confirmed and not a.evaluator_call_id,
              'DGM_RECOVERY_CONFIRMED_OR_TERMINAL')
        usage_path = run.directory / 'codex_calls' / (a.call_id + '.json')
        usage = io._read_json(usage_path) if usage_path.exists() else None
        check(usage is None or (usage.get('call_id') == a.call_id
              and usage.get('started_calls') == 0 and usage.get('successful_calls') == 0),
              'DGM_RECOVERY_CONFIRMED_OR_TERMINAL')
        prepared, work = owned_resource(run.ledger, plan, a.attempt_id)
        from downstream_lifecycle import workspace_fingerprint
        check(workspace_fingerprint(work / 'testbed') == prepared.workspace_fingerprint,
              'DGM_RECOVERY_WORKSPACE_CHANGED')
        body = {'schema_version': 1, 'reason': REASON, 'run_directory': str(run.directory),
            'run_id': run.manifest['run_id'], 'run_fingerprint': plan.run_fingerprint,
            'checkpoint_digest': checkpoint['sha256'], 'plan_digest': io.digest(asdict(plan)),
            'evaluation_id': evaluation_id, 'task_id': plan.identity.task_id,
            'attempt_id': a.attempt_id, 'new_attempt_id': f'attempt-{len(snap.attempts)+1:06d}',
            'head': snap.events[-1].sha256, 'usage_digest': io.digest(usage),
            'implementation': implementation_row}
        value = {'authorization': body, 'sha256': io.digest(body)}
        path = run.ledger._folder(evaluation_id) / 'artifacts' / a.attempt_id / 'operator-waiver.json'
        if path.exists():
            check(io._read_json(path) == value, 'DGM_RECOVERY_AUTHORIZATION_CHANGED')
        else:
            durable_json(path, value)
        reference = io.Artifact(str(path.relative_to(run.ledger._folder(evaluation_id))),
                                hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_size)
        with run.ledger.session(plan.identity) as session:
            check(session.snapshot() == snap, 'DGM_RECOVERY_STATE_CHANGED')
            session._append('attempt_abandoned', a.attempt_id,
                            {'artifacts': [asdict(reference)], 'reason': REASON})
        record_implementation(run, implementation_row)
        return {'status': 'authorized', 'token': value['sha256'], 'evaluation_id': evaluation_id,
                'next_attempt': body['new_attempt_id'], 'model_calls': 0, 'evaluator_calls': 0,
                'historical_outcome': 'unknown_not_a_failed_observation'}


def cleanup(directory, *, client=None):
    """Remove waived resources; stop resumable prepared resources, preserving work."""
    run, _ = open_run(directory)
    with run.lock():
        selected = []
        for plan in run.plans.read():
            inspected = run.ledger.inspect(plan.identity.evaluation_id, plan.identity)
            check(not io.classify(inspected).value.startswith('BLOCK_'), 'DGM_CLEANUP_BLOCKED')
            snap = inspected.snapshot
            if not snap or not snap.attempts:
                continue
            a = snap.attempts[-1]
            if a.state == 'attempt_abandoned':
                authorized_spec(run.ledger, plan, inspected, require_cleanup=False)
            if a.cleanup_status == 'success' or a.state not in ('attempt_abandoned', 'preparation_terminal'):
                continue
            check(a.preparation_status == 'success' and not a.model_start_confirmed
                  and not a.evaluator_call_id, 'DGM_CLEANUP_BLOCKED')
            prepared, work = owned_resource(run.ledger, plan, a.attempt_id)
            selected.append((plan, a, prepared, work))
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
                except NotFound:
                    # A prior removal may have completed before cleanup's ledger
                    # publication. Only an audited waiver plus durable ownership
                    # intent permits this local, zero-call reconciliation.
                    row = io._read_json(work / 'resource-suspension.json')
                    identity = row['identity']
                    check(a.state == 'attempt_abandoned' and row['sha256'] == io.digest(identity)
                          and identity['id'] == prepared.resource_id
                          and identity['context'] == plan.spec(a.attempt_id).preparation_envelope()
                          and identity['name'] == 'sweb.eval.' + plan.identity.task_id + '.' + plan.identity.evaluation_id,
                          'DGM_RESOURCE_OWNERSHIP')
                    verified.append((plan, a, work, None, identity))
                    continue
                identity = container_identity(container, plan.spec(a.attempt_id), prepared)
                verified.append((plan, a, work, container, identity))
            # Validate every ownership before stopping/removing the first resource.
            for plan, a, work, container, identity in verified:
                receipt = work / 'resource-suspension.json'
                row = {'schema_version': 1, 'identity': identity, 'sha256': io.digest(identity)}
                if receipt.exists():
                    check(io._read_json(receipt) == row, 'DGM_RESOURCE_SUSPENSION_CHANGED')
                else:
                    durable_json(receipt, row)
                if container is not None and container.status == 'running':
                    container.stop(timeout=10)
                if a.state == 'attempt_abandoned':
                    if container is not None:
                        container.remove()
                    with run.ledger.session(plan.identity) as session:
                        session.finish_cleanup(a.attempt_id)
            return {'removed': sum(a.state == 'attempt_abandoned' for _, a, _, _, _ in verified),
                    'suspended': sum(a.state != 'attempt_abandoned' for _, a, _, _, _ in verified)}
        finally:
            if external:
                client.close()


def resume_resource(container, spec, prepared, work):
    path = work / 'resource-suspension.json'
    if not path.exists():
        return
    row = io._read_json(path)
    identity = container_identity(container, spec, prepared)
    check(row == {'schema_version': 1, 'identity': identity, 'sha256': io.digest(identity)},
          'DGM_RESOURCE_SUSPENSION_CHANGED')
    if container.status == 'exited':
        container.start()
    else:
        check(container.status == 'running', 'DGM_RESOURCE_STATE')
