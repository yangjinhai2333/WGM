"""Bounded execution of already independent plans; ledger remains the authority.

The caller owns the existing run lease. Only the controller publishes plans and
consumes results, in plan order. Workers own single-task lifecycles, no RNG or
algorithm state. Runtime JSON files below are disposable accounting, not recovery.
"""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager, nullcontext
import signal
import threading
import time

import evaluation_ledger as io
from evaluation_plan import check, durable_json
from evaluation_scheduler import read_binding, SchedulingBlocked
from cancellation import request as cancel_run, primary, safe_error


def runtime_workers(config, run_type, count=1, *, domain=None):
    from config import validate_evaluation_workers
    configured = validate_evaluation_workers(config.execution.evaluation_workers)
    eligible = config.execution.policy == 'validated' and (
        run_type in ('single-task', 'small-baseline', 'default-baseline') or domain in ('primary', 'secondary'))
    return {'configured_workers': configured, 'effective_workers': min(configured, max(1, count)) if eligible else 1,
            'scope': 'independent_fixed_evaluations' if eligible else 'serial_entry',
            'scientific_identity': False}


def batch_reports(directory):
    return [io._read_json(p) for p in sorted((directory / 'evaluation_batches').glob('*.json'))]


def independent_plan(run, plan):
    if plan.identity.stage in ('root', 'final_benchmark'):
        return True
    if plan.identity.stage != 'search' or run.manifest.get('experiment', {}).get('algorithm') != 'dgm':
        return False
    # DGM stages are fixed before this wrapper. Adaptive WGM/HGM remain forbidden.
    from dgm_legacy_search import read_checkpoint
    state = read_checkpoint(run.directory, run.manifest['reproducibility_fingerprint'])['state']
    from dataclasses import asdict
    return any(asdict(plan) == p for g in state['generations'] for child in g['children']
               for stage in child['stages'].values() for p in stage['plans'])


@contextmanager
def _interrupts(cancel):
    """First Ctrl-C stops submissions/readiness; running model calls can reap."""
    old = {}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGINT, signal.SIGTERM):
            old[sig] = signal.getsignal(sig)
            signal.signal(sig, lambda signum, frame: cancel_run(cancel, signum=signum))
    try:
        yield
    finally:
        for sig, handler in old.items():
            signal.signal(sig, handler)


def _ledger_counts(run, plans):
    counts = {'downstream_model': 0, 'final_model': 0, 'evaluator': 0, 'diagnosis': 0, 'generation': 0}
    completed = failed = ambiguous = 0
    results = []
    for plan in plans:
        inspection = run.ledger.inspect(plan.identity.evaluation_id, plan.identity)
        check(not inspection.error_code, 'BATCH_LEDGER_INVALID')
        snapshot = inspection.snapshot
        if snapshot:
            for event in snapshot.events:
                if event.kind == 'model_start_confirmed':
                    counts['final_model' if plan.identity.stage == 'final_benchmark' else 'downstream_model'] += 1
                elif event.kind == 'evaluation_start_confirmed':
                    counts['evaluator'] += 1
            if snapshot.state == 'bound':
                result = read_binding(run.ledger, plan)
                completed += result.outcome == 'success'
                failed += result.outcome == 'failed'
                results.append({'evaluation_id': plan.identity.evaluation_id, 'task_id': plan.identity.task_id,
                                'outcome': result.outcome, 'resolved': result.resolved})
            elif io.classify(inspection).value == 'RETRY_PREPARATION':
                failed += 1
            ambiguous += 'AMBIGUOUS' in io.classify(inspection).value
    return {'completed': completed, 'failed': failed, 'incomplete': len(plans)-completed-failed,
            'ambiguous': ambiguous, 'stage_calls': counts, 'results': results}


class SharedEvaluationPool:
    """One run's task capacity, shared by overlapping fixed child batches.

    The owner holds the run lease and enters before starting child workers. The
    full recovery precheck therefore sees no live markers. Later batches still
    validate their own plans/bindings; each lifecycle checks its own ledger.
    The publication lock is held only for plan/runtime publication, never calls.
    """
    def __init__(self, run, workers, cancel_event, publication_lock):
        from config import validate_evaluation_workers
        self.workers = validate_evaluation_workers(workers)
        self.run, self.cancel, self.publication_lock = run, cancel_event, publication_lock
        self.pool = None
        self.active = self.maximum = 0
        self.mutex = threading.Lock()

    def __enter__(self):
        self.run.precheck()
        self.pool = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix='run-evaluation')
        return self

    def __exit__(self, kind, value, traceback):
        if kind is not None:
            cancel_run(self.cancel, value)
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.pool = None

    @contextmanager
    def running(self):
        with self.mutex:
            self.active += 1
            self.maximum = max(self.maximum, self.active)
        try:
            yield
        finally:
            with self.mutex:
                self.active -= 1


def execute_batch(run, plans, execute, *, workers=1, consume=lambda plan, result: None,
                  cancel_event=None, shared_pool=None):
    """Freeze the entire ordered batch, then execute at most workers tasks.

    Stop replenishing on failure/cancel, drain started tasks, retain every binding.
    Only the contiguous successful prefix is consumed. Resume skips all bound IDs,
    including successes after a failed/gapped prefix. Ambiguous calls never retry.
    """
    from config import validate_evaluation_workers
    validate_evaluation_workers(workers)
    plans = tuple(plans)
    check(bool(plans) and len({p.identity.evaluation_id for p in plans}) == len(plans)
          and list(plans) == sorted(plans, key=lambda p: p.identity.ordinal)
          and all(independent_plan(run, p) for p in plans), 'BATCH_NOT_INDEPENDENT')
    if shared_pool is not None:
        check(shared_pool.run is run and shared_pool.workers == workers and shared_pool.pool is not None
              and shared_pool.cancel is cancel_event, 'BATCH_SHARED_POOL_MISMATCH')
    batch_id = io.digest([p.identity.evaluation_id for p in plans])
    folder = run.directory / 'evaluation_batches'
    folder.mkdir(exist_ok=True)
    path = folder / (batch_id + '.json')
    previous = io._read_json(path) if path.exists() else {'sessions': []}
    effective = min(workers, len(plans))
    runtime = {'configured_workers': workers, 'effective_workers': effective,
               'scope': 'independent_fixed_evaluations', 'scientific_identity': False}
    if shared_pool is not None:
        runtime['capacity_scope'] = 'run_global'
    with shared_pool.publication_lock if shared_pool is not None else nullcontext():
        run.verify()
        published = {p.identity.evaluation_id: p for p in run.plans.read()}
        for plan in plans:
            previous_plan = published.get(plan.identity.evaluation_id)
            check(previous_plan is None or previous_plan == plan, 'BATCH_PLAN_CHANGED')
            inspection = run.ledger.inspect(plan.identity.evaluation_id, plan.identity)
            check(not inspection.error_code, 'BATCH_LEDGER_INVALID')
            if previous_plan is None or inspection.snapshot is None:
                run.publish(plan)  # All durable before worker startup, including on serial runs.
        history = run.manifest.get('evaluation_parallelism', {}).get('history', [])
        run.manifest['evaluation_parallelism'] = {**runtime, 'history': [*history, {**runtime, 'batch_id': batch_id}]}
        durable_json(run.directory / 'run_manifest.json', run.manifest, replace_existing=True)
    cancel = cancel_event if cancel_event is not None else threading.Event()
    started = time.monotonic()
    active = maximum = submitted = skipped = 0
    durations, values, errors = {}, {}, {}
    mutex = threading.Lock()
    fatal = None
    next_consume = 0
    stats = {'schema_version': 1, 'batch_id': batch_id, 'plan_ids': [p.identity.evaluation_id for p in plans],
             **runtime, 'requested': len(plans), 'status': 'running', 'sessions': previous['sessions']}
    durable_json(path, stats, replace_existing=True)

    def task(index):
        nonlocal active, maximum
        if cancel.is_set():
            raise KeyboardInterrupt
        tick = time.monotonic()
        with mutex:
            active += 1
            maximum = max(maximum, active)
        try:
            with shared_pool.running() if shared_pool is not None else nullcontext():
                return run.execute(plans[index], execute, independent=True)
        except BaseException as error:
            if shared_pool is not None or cancel.is_set():
                cancel_run(cancel, error)
            raise
        finally:
            with mutex:
                active -= 1
                durations[index] = time.monotonic() - tick

    def collect_prefix():
        nonlocal next_consume
        while next_consume in values and next_consume not in errors and not cancel.is_set():
            consume(plans[next_consume], values[next_consume])
            next_consume += 1

    with _interrupts(cancel):
        pool = None
        pending = {}
        try:
            if shared_pool is None:
                run.precheck()  # Shared pools precheck once, before either child starts.
            todo = []
            for i, plan in enumerate(plans):
                snap = run.ledger.inspect(plan.identity.evaluation_id, plan.identity).snapshot
                if snap and snap.state == 'bound':
                    values[i] = read_binding(run.ledger, plan)
                    skipped += 1
                    if values[i].outcome != 'success':
                        errors[i] = SchedulingBlocked('BOUND_FAILURE', plan)
                else:
                    todo.append(i)
            collect_prefix()
            pool = (shared_pool.pool if shared_pool is not None else
                    ThreadPoolExecutor(max_workers=effective, thread_name_prefix='fixed-evaluation'))
            cursor = 0
            while pending or (cursor < len(todo) and not errors and not cancel.is_set()):
                while len(pending) < effective and cursor < len(todo) and not errors and not cancel.is_set():
                    index = todo[cursor]
                    pending[pool.submit(task, index)] = index
                    cursor += 1
                    submitted += 1
                if not pending:
                    break
                done, _ = wait(pending, timeout=.1, return_when=FIRST_COMPLETED)
                for future in sorted(done, key=lambda f: pending[f]):
                    index = pending.pop(future)
                    try:
                        value = values[index] = future.result()
                        if value.outcome != 'success':
                            errors[index] = SchedulingBlocked('BOUND_FAILURE', plans[index])
                    except BaseException as error:
                        errors[index] = error
                collect_prefix()
        except BaseException as error:
            fatal = error
            cancel_run(cancel, error)
        finally:
            if pool is not None:
                for future in pending:
                    future.cancel()
                if shared_pool is None:
                    pool.shutdown(wait=True, cancel_futures=True)
                elif pending:
                    wait(pending)  # Drain this batch; the sibling still owns its running calls.
                for future, index in pending.items():
                    if not future.cancelled():
                        try:
                            values[index] = future.result()
                        except BaseException as error:
                            errors[index] = error
            counts = _ledger_counts(run, plans)
            session = {**runtime, 'submitted': submitted, 'skipped_bound': skipped,
                       'batch_wall_seconds': time.monotonic()-started, 'task_duration_sum_seconds': sum(durations.values()),
                       'observed_max_concurrency': maximum, 'interrupted': cancel.is_set()}
            session['cancellation_origin'] = getattr(cancel, '_wgm_cancel_origin', None)
            session['errors'] = [{'evaluation_id': plans[i].identity.evaluation_id,
                                 **safe_error(e)} for i, e in sorted(errors.items())]
            session['exit_states'] = []
            for plan in plans:
                inspected = run.ledger.inspect(plan.identity.evaluation_id, plan.identity)
                snap = inspected.snapshot
                session['exit_states'].append({'evaluation_id': plan.identity.evaluation_id,
                    'decision': io.classify(inspected).value,
                    'durable_head': snap.events[-1].sha256 if snap else None,
                    'attempt_id': snap.attempts[-1].attempt_id if snap and snap.attempts else None})
            if shared_pool is not None:
                session['observed_run_max_concurrency'] = shared_pool.maximum
            sessions = [*previous['sessions'], session]
            stats.update(counts, **session, sessions=sessions,
                         cumulative_batch_wall_seconds=sum(s['batch_wall_seconds'] for s in sessions),
                         cumulative_task_duration_sum_seconds=sum(s['task_duration_sum_seconds'] for s in sessions),
                         peak_concurrency=max(s['observed_max_concurrency'] for s in sessions),
                         status='completed' if counts['completed'] == len(plans) and not fatal and not errors
                         and not cancel.is_set() else 'incomplete')
            durable_json(path, stats, replace_existing=True)
    error = primary(cancel, fatal or (errors[min(errors)] if errors else None))
    if error is not None:
        raise error
    if cancel.is_set():
        raise KeyboardInterrupt
    return [values[i] for i in range(len(plans))]


def read_fixed_report(directory, manifest):
    """Baseline report from existing bindings; no legacy task metadata required."""
    from evaluation_scheduler import ExecutionRun
    run = ExecutionRun(directory, manifest)
    plans = run.plans.read()
    expected = {p.identity.evaluation_id: p.identity for p in plans}
    for snapshot in run.ledger.evaluations():
        check(expected.get(snapshot.identity.evaluation_id) == snapshot.identity, 'REPORT_UNCLAIMED_EVALUATION')
    counts = _ledger_counts(run, plans)
    summary = io._read_json(directory / 'run_summary.json')
    return {'report_type': 'fixed_evaluation_batch', 'status': summary['status'],
            'run_type': manifest['run_type'], 'commit': manifest['repository']['commit'],
            'execution_policy': manifest['execution_policy']['effective_policy'],
            'requested': len(plans), **counts, 'resolved': sum(r['resolved'] is True for r in counts['results']),
            'evaluation_parallelism': manifest.get('evaluation_parallelism'),
            'evaluation_batches': batch_reports(directory),
            'timing': summary['timing'], 'network_readiness': summary.get('network_readiness')}


def fixed_report_text(summary):
    import json
    return '\n'.join([
        f"{summary['run_type']}: {summary['status']}; policy={summary['execution_policy']}",
        f"Run commit: {summary['commit']}",
        f"Requested/completed/failed/incomplete/resolved: {summary['requested']}/{summary['completed']}/{summary['failed']}/{summary['incomplete']}/{summary['resolved']}",
        'Evaluation workers: ' + json.dumps(summary['evaluation_parallelism'], sort_keys=True),
        'Stage calls: ' + json.dumps(summary['stage_calls'], sort_keys=True),
        *('Fixed evaluation batch: ' + json.dumps(row, sort_keys=True) for row in summary['evaluation_batches'])])
