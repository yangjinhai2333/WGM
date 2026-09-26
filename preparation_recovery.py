"""Dispatch of explicit, audited retry grants for one evaluation plan; no search recovery.

Retries are never inferred from a failure. A failed attempt continues only through a
durable grant written by one of the explicit authorization commands:
final readiness (``final_readiness_recovery``), connection-failure search evaluations
(``evaluation_retry``), DGM interruptions (``dgm_recovery``) and failed final
preparation (``preparation_authorization``). Resume requires the run's own commit.
"""
import evaluation_ledger as io
from evaluation_plan import check

REASON = 'final_scheduler_consumes_audited_retry_preparation_v1'


def _failure(snapshot, attempt_id):
    attempts = [a for a in snapshot.attempts if a.attempt_id == attempt_id]
    check(len(attempts) == 1, 'PREPARATION_RETRY_ATTEMPT')
    attempt = attempts[0]
    events = [e for e in snapshot.events if e.attempt_id == attempt_id]
    check(attempt.preparation_status == 'failed' and attempt.cleanup_status == 'success'
          and attempt.call_id is None and attempt.evaluator_call_id is None
          and not any(e.kind.startswith(('model_', 'evaluation_')) for e in events),
          'PREPARATION_RETRY_UNSAFE')
    terminal = next(e for e in events if e.kind == 'preparation_terminal')
    cleanup = next(e for e in events if e.kind == 'cleanup_terminal')
    check(io._decode(terminal.data)['status'] == 'failed'
          and io._decode(cleanup.data)['status'] == 'success', 'PREPARATION_RETRY_UNSAFE')
    return terminal.sha256, cleanup.sha256


def execution_spec(ledger, plan, inspection):
    """Read-only validation of a durable grant; never infer permission from failure."""
    from evaluation_scheduler import SchedulingBlocked
    check(not inspection.error_code and not inspection.legacy and not inspection.read_only,
          'PREPARATION_RETRY_LEDGER')
    if plan.identity.stage == 'final_benchmark' and inspection.snapshot and inspection.snapshot.attempts:
        from final_readiness_recovery import execution_spec as readiness_spec
        spec = readiness_spec(ledger, plan, inspection)
        if spec is not None:
            return spec
    if inspection.snapshot and any(a.authorization_artifacts for a in inspection.snapshot.attempts):
        from evaluation_retry import applies, authorized_spec as connection_spec
        if applies(inspection.snapshot):
            return connection_spec(ledger, plan, inspection)
        from dgm_recovery import authorized_spec
        return authorized_spec(ledger, plan, inspection)
    from preparation_authorization import records, execution_spec as authorized_spec
    grants = records(ledger, plan)
    if grants:
        return authorized_spec(ledger, plan, inspection, grants)
    if io.classify(inspection, plan.identity).value == 'RETRY_PREPARATION':
        raise SchedulingBlocked('FAILED_PREPARATION', plan)
    return plan.spec()


def authorize(run, token, implementation):
    """Caller holds the run lease. Bind one explicit audit token to one new attempt."""
    io._sha(token)
    from preparation_authorization import use_token
    result = use_token(run, token, implementation)
    check(result is not None, 'PREPARATION_AUTHORIZATION_TOKEN')
    return result


def implementation_identity(root, old_commit, repository, **_):
    """Final evaluation and its retries run on the exact commit that produced the run."""
    from run_metadata import check_repository
    check_repository(repository)
    check(repository['commit'] == old_commit, 'FINAL_COMMIT_CHANGED')
    return {'old_commit': old_commit, 'new_commit': repository['commit'], 'reason': REASON}


def check_implementation(run, root, old_commit, repository, token, **_):
    """No grant is consumed until the commit and every retry precondition are validated."""
    identity = implementation_identity(root, old_commit, repository)
    if token:
        authorize(run, token, identity)
    run.precheck()  # Includes every ambiguous call and every unsafe preparation.
    return identity
