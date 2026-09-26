"""Serial V2 controller over N1/N2 and the existing validated evaluation ledger."""
from collections import defaultdict
from copy import copy, deepcopy
from dataclasses import asdict
import json
import math
from pathlib import Path
import warnings

import evaluation_ledger as io
from evaluation_plan import EvaluationPlan, check, durable_json
from evaluation_scheduler import ExecutionRun, SchedulingBlocked, read_binding
from search_operation_journal import durable_record
from search_operations import SearchOperations, read_checkpoint
from search_state import save_checkpoint
from validated_search import ValidatedSearch
from wgm_v2_config import CHECKPOINTS, SPEC, pause_target, runtime_control, resolve_initial_source
from wgm_v2_state import WGMV2State


class V2Operations(SearchOperations):
    def commit(self, engine, plan, terminal):
        # Diagnosis exhaustion is a trusted model-format failure, not evidence
        # that a generated child is non-executable. It supplies no EXP3 reward.
        if terminal['outcome'] == 'diagnosis_invalid':
            raise SchedulingBlocked('V2_DIAGNOSIS_OUTPUT_INVALID_EXHAUSTED')
        return super().commit(engine, plan, terminal)


class WGMV2Search(ValidatedSearch):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        check(self.config.search.algorithm == 'wgm_v2' and self.operations is not None, 'V2_ENGINE_IDENTITY')

    def save(self):
        save_checkpoint(self.run.directory, self.state, self.count, self.fingerprint, fault=self.fault)

    def restore(self):
        checkpoint = read_checkpoint(self.run.directory)
        super().restore()
        self.state.runtime_control = deepcopy(checkpoint['runtime_control'])

    def transaction(self, change):
        try:
            change()
            self.save()
        except BaseException:
            self.restore()
            raise

    def set_target(self, target):
        check(type(target) is int and (target in CHECKPOINTS or
              self.config.protocol.effective_name == 'smoke' and 1 <= target <= 800), 'V2_TARGET_INVALID')
        saved_control = self.run.manifest.get('wgm_v2_runtime', self.state.runtime_control)
        control = deepcopy(saved_control)
        check(control['protocol'] == self.config.protocol.effective_name, 'V2_PROTOCOL_CHANGED')
        control['zero_progress_expansion_cap'] = self.config.wgm_v2.safety_attempt_cap
        old = control['target_budget']
        check(target >= old, 'V2_TARGET_DECREASE')
        if target != old:
            check(old in CHECKPOINTS and target in CHECKPOINTS and
                  CHECKPOINTS.index(target) == CHECKPOINTS.index(old)+1 and self.count == old,
                  'V2_CONTINUE_REQUIRES_COMPLETED_PREVIOUS_TARGET')
            control['continuations'].append({'from': old, 'to': target, 'valid_observations': self.count,
                'canonical_checkpoint_digest': read_checkpoint(self.run.directory)['sha256']})
            control['target_budget'] = target
        self.state.runtime_control = control
        proposed = deepcopy(self.run.manifest)
        proposed['wgm_v2_runtime'] = deepcopy(control)
        if not io.same_json(proposed, self.run.manifest):
            durable_json(self.run.directory / 'run_manifest.json', proposed, replace_existing=True)
            self.run.manifest.clear()
            self.run.manifest.update(proposed)
        self.save()

    def _initialized_manifest(self):
        if not self.state.initialized:
            return
        value = {'run_fingerprint': self.fingerprint, **self.state.initialized_identity}
        path = self.run.directory / 'wgm_v2_initialized.json'
        if path.exists():
            check(io.same_json(io._read_json(path), value), 'V2_INITIALIZED_IDENTITY')
        else:
            durable_record(path, value)

    def milestone(self):
        if self.count not in CHECKPOINTS:
            return
        # No adaptive action occurs between result consumption and this snapshot.
        saved = read_checkpoint(self.run.directory)
        folder = self.run.directory / 'checkpoints'
        folder.mkdir(exist_ok=True)
        path = folder / f'{self.count:06d}.json'
        row = {'schema_version': 1, 'domain': 'canonical', 'b': self.count,
               'run_fingerprint': self.fingerprint, 'state_digest': saved['sha256'],
               'state': saved['state'], 'runtime_control': saved['runtime_control'],
               'search_wall_time_seconds': recorded_wall_time(self.operations.recorder)}
        if path.exists():
            previous = io._read_json(path)
            check(previous['state_digest'] == row['state_digest'] and previous['state'] == row['state'],
                  'V2_CHECKPOINT_CONFLICT')
        else:
            durable_record(path, row)
        selection = {**self.state.top_lineages(), 'canonical_checkpoint_digest': saved['sha256'],
                     'valid_search_evaluations': self.count,
                     'derived_bai_budget': math.floor(self.config.wgm_v2.checkpoint_bai_fraction*self.count),
                     'derived_bai_status': 'not_executed'}
        selection_path = folder / f'{self.count:06d}.selection.json'
        if selection_path.exists():
            check(io.same_json(io._read_json(selection_path), selection), 'V2_SELECTION_CONFLICT')
        else:
            durable_record(selection_path, selection)

    def reconcile(self):
        self._validate_consumption()
        pending = self.state.inflight
        if pending and pending['kind'] == 'evaluation':
            self.run.publish(EvaluationPlan.from_dict(pending['evaluation_plan']))
        self.run.precheck(self.state.execution['consumed_bindings'])
        self.operations.reconcile(self)
        if self.state.inflight and self.state.inflight['kind'] == 'evaluation':
            self._finish_evaluation(EvaluationPlan.from_dict(self.state.inflight['evaluation_plan']))
        planned = {p.identity.evaluation_id for p in self.run.plans.read()}
        check(planned == set(self.state.execution['consumed_bindings']), 'V2_ORPHAN_EVALUATION')
        self._initialized_manifest()
        self.milestone()

    def _new_round(self):
        state, cfg = self.state, self.config.wgm_v2
        initial = not state.initialized
        eid = f'expansion-{len(state.history):06d}'
        available = cfg.planning_horizon-self.count
        if available < cfg.M:
            self.transaction(lambda: setattr(state, 'stop_reason', 'insufficient_full_batch_budget'))
            return
        if not initial and state.zero_progress_expansions >= cfg.safety_attempt_cap:
            self.transaction(lambda: setattr(state, 'stop_reason', 'zero_progress_expansion_cap'))
            return
        # A frozen founder may not silently disappear or be replaced.
        for candidate in state.founders.values():
            self.operations.sources.get(candidate)
        def create():
            slots = state.reserve(self.run.manifest['run_id'], eid,
                                  'initialization' if initial else 'child_batch', cfg.M)
            selected = None if initial else state.selector.select(eid)
            state.round = {'expansion_id': eid, 'initial': initial, 'batch_slots': slots,
                'selected': selected, 'start_count': self.count, 'local_limit': 0 if initial else min(cfg.local_bai_evals, available-cfg.M),
                'local_completed': 0, 'phase': 'generation' if initial else 'local',
                'parent_id': 0 if initial else None, 'parent_draw': None, 'child_id': None}
        self.transaction(create)

    def _select_expansion(self):
        state, current = self.state, self.state.round
        check(current is not None and current['phase'] == 'generation', 'V2_ROUND_PHASE')
        parent = state.nodes[current['parent_id']]
        index = len(state.history)
        cid = f'candidate-{index:06d}'
        check(current['expansion_id'] == f'expansion-{index:06d}', 'V2_EXPANSION_ID')
        state.history.append({'expansion_id': current['expansion_id'], 'candidate_id': cid,
            'parent_id': parent.id, 'selected_branch_id': state.branches[parent.id],
            'selection_probability': None if current['initial'] else current['selected']['probability'],
            'child_id': None, 'phase': 'initialization' if current['initial'] else 'search',
            'target_M': self.config.wgm_v2.M, 'status': 'creating'})
        state.inflight = {'kind': 'expansion', 'expansion_id': current['expansion_id']}
        return parent, cid

    def _local_or_parent(self):
        state, current, cfg = self.state, self.state.round, self.config.wgm_v2
        branch = current['selected']['lineage_id']
        choice = state.choose_bai(branch)
        if current['local_completed'] < current['local_limit'] and choice['evaluate'] is not None:
            candidate = state.nodes[choice['evaluate']['node_id']]
            # Child slots were reserved first; task draws follow actual execution order.
            self._evaluate(candidate, domain='local_bai')
            return
        def parent():
            draw = state.selection_rng.random()  # consumed even when leader == founder
            cid = state.founders[branch] if draw < cfg.rho_r else choice['leader']['candidate_id']
            node = next(n for n in state.nodes.values() if n.commit_id == cid)
            current.update(parent_draw=draw, parent_id=node.id, phase='generation',
                           local_stop_reason=choice['stop_reason'] or 'budget_exhausted')
        self.transaction(parent)

    def _evaluate(self, node, *, domain, slot_id=None):
        state = self.state
        def plan_evaluation():
            sid = slot_id or state.reserve(self.run.manifest['run_id'], state.round['expansion_id'], domain, 1)[0]
            slot = state.slots[sid]
            check(slot['status'] == 'reserved', 'V2_SLOT_REUSE')
            task = state.tasks[state.task_rng.randrange(len(state.tasks))]
            draw = len(state.draws)
            plan = self.make_plan(node, task, draw)
            # The immutable ledger ID binds the reservation, operation, candidate and task.
            plan = EvaluationPlan.create(run_id=plan.identity.run_id, stage='search', candidate_id=node.commit_id,
                entry=self.operations.entries[task], ordinal=plan.identity.ordinal, task_draw_index=draw,
                run_fingerprint=plan.run_fingerprint, task_schedule_identity=plan.task_schedule_identity,
                policy_fingerprint=plan.identity.policy_fingerprint, source_fingerprint=plan.identity.source_fingerprint,
                model_fingerprint=plan.identity.model_fingerprint, evaluator_fingerprint=plan.identity.evaluator_fingerprint,
                search_identity=plan.search_identity, selection_digest=io.digest({
                    'reservation_id': sid, 'expansion_id': state.round['expansion_id'],
                    'operation_id': next((p.operation_id for p in self.operations.journal.plans()
                         if p.identity['candidate_id'] == node.commit_id), None)}))
            slot.update(status='sampled', candidate_id=node.commit_id, task_id=task,
                        task_draw_index=draw, evaluation_id=plan.identity.evaluation_id)
            state.draws.append({'ordinal': draw, 'task_id': task, 'candidate_id': node.commit_id,
                                'reservation_id': sid, 'evaluation_id': plan.identity.evaluation_id})
            state.inflight = {'kind': 'evaluation', 'node_id': node.id, 'task_id': task,
                              'slot_id': sid, 'evaluation_plan': asdict(plan)}
        self.transaction(plan_evaluation)
        self.fault('v2.evaluation.after_plan_checkpoint')
        plan = EvaluationPlan.from_dict(state.inflight['evaluation_plan'])
        self.run.publish(plan)
        self._finish_evaluation(plan)

    def _finish_evaluation(self, plan):
        result = self.run.execute(plan, self.execute_one)
        if result.outcome != 'success':
            raise SchedulingBlocked('BOUND_FAILURE', plan)
        check(read_binding(self.run.ledger, plan) == result, 'V2_BINDING_CHANGED')
        self.fault('v2.binding.before_consumption')
        def consume():
            if self.state.observe_binding(result):
                self.count += 1
        self.transaction(consume)
        self.fault('v2.binding.after_checkpoint')
        self.milestone()
        self.refresh_views()

    def run_to_budget(self):
        self.reconcile()
        target = self.state.runtime_control['target_budget']
        while self.count < target and self.state.stop_reason is None:
            state = self.state
            if not state.initialized and state.round is None and state.initial_attempts == self.config.wgm_v2.initial_expansion_target:
                self.transaction(state.freeze)
                self._initialized_manifest()
                continue
            if state.round is None:
                self._new_round()
                continue
            phase = state.round['phase']
            if phase == 'local':
                self._local_or_parent()
            elif phase == 'generation':
                self._expand()  # Existing N2 plan -> gate -> marker -> terminal -> admission.
            elif phase == 'batch':
                sid = next(s for s in state.round['batch_slots'] if state.slots[s]['status'] == 'reserved')
                self._evaluate(state.nodes[state.round['child_id']], domain=state.slots[sid]['domain'], slot_id=sid)
            else:
                raise ValueError('V2_ROUND_PHASE')
        self.milestone()
        return self.count

    def write_outputs(self, error=None):
        stats = physical_accounting(self.run, self.operations)
        self.state.runtime_control['accounting'] = stats
        self.save()
        summary = {**self.state.summary(), **stats, 'runtime_control': deepcopy(self.state.runtime_control),
            'status': 'incomplete' if error or self.count < self.state.runtime_control['target_budget'] else 'paused',
            'error_code': error, 'planning_horizon': 800,
            'formal_comparison_eligible': self.run.manifest['experiment']['requested_execution_policy'] == 'validated'
                and self.state.runtime_control['protocol'] == 'formal',
            'protocol': self.state.runtime_control['protocol'],
            'smoke_coverage': smoke_coverage(self.state),
            'search_benchmark_overlap': self.run.manifest.get('v2_sampling', {}).get('benchmark_overlap'),
            'production_ready': False, 'checkpoint_bai_status': 'explicit_only', 'final_benchmark_status': 'not_executed'}
        durable_json(self.run.directory / 'wgm_v2_summary.json', summary, replace_existing=True)
        durable_json(self.run.directory / 'wgm_v2_selection.json',
                     {**self.state.top_lineages(), 'canonical_state_digest': read_checkpoint(self.run.directory)['sha256']},
                     replace_existing=True)
        return summary


def physical_accounting(run, operations):
    """Derived from the existing two journals, never authorization or rewards."""
    result = {k: 0 for k in ('physical_model_calls', 'physical_evaluator_calls', 'diagnosis_calls',
        'generation_calls', 'downstream_model_calls', 'model_markers', 'evaluator_markers',
        'failed_evaluations', 'ambiguous_evaluations', 'infrastructure_operations',
        'ambiguous_operations', 'model_output_invalid_attempts')}
    charged = 0
    for snapshot in run.ledger.evaluations():
        events = [e.kind for e in snapshot.events]
        result['model_markers'] += events.count('model_started')
        result['evaluator_markers'] += events.count('evaluation_started')
        result['downstream_model_calls'] += events.count('model_start_confirmed')
        result['physical_evaluator_calls'] += events.count('evaluation_start_confirmed')
        charged += int('model_start_confirmed' in events)
        decision = io.classify(run.ledger.inspect(snapshot.identity.evaluation_id)).value
        result['ambiguous_evaluations'] += int('AMBIGUOUS' in decision)
        latest = next((io._decode(e.data) for e in reversed(snapshot.events) if e.kind == 'bound'), None)
        result['failed_evaluations'] += int(decision == 'RETRY_PREPARATION' or (
            latest is not None and snapshot.state == 'bound' and latest['outcome'] == 'failed'))
    for plan in operations.journal.plans():
        state, _ = operations.journal.inspect(plan)
        result['infrastructure_operations'] += int(state['blocked'] is not None)
        for stage in ('diagnosis', 'generation'):
            for attempt in state[stage]:
                result[stage+'_calls'] += int(attempt.get('started', False))
                result['model_markers'] += int('marker' in attempt)
                result['ambiguous_operations'] += int('marker' in attempt and 'terminal' not in attempt)
                result['infrastructure_operations'] += int('terminal' in attempt and
                    not __import__('search_operation_journal').successful(attempt['terminal']['evidence']))
                result['model_output_invalid_attempts'] += int(attempt.get('result', {}).get('outcome') == 'invalid')
    result['physical_model_calls'] = result['diagnosis_calls']+result['generation_calls']+result['downstream_model_calls']
    result['charged_evaluation_slots'] = charged
    result['physical_count_definition'] = 'local_process_start_confirmations_not_remote_API_requests'
    return result


def diagnosis_entry(engine, parent):
    completed = [s for s in engine.state.slots.values() if s.get('candidate_id') == parent and s['status'] == 'completed']
    if not completed:
        return 'source_only'
    failed = [s for s in completed if s['observation'] == 0]
    return max(failed or completed, key=lambda s: s['task_draw_index'])['task_id']


def diagnosis_prompt(config, sources, directory, entries, entry, parent):
    from diagnosis_context import bounded_prompt
    source = sources.get(parent)
    parts = {'policy': source['policy'], 'task': 'Source/policy-only initialization; no task observations.',
             'problem': 'Propose one general improvement without assuming benchmark outcomes.'}
    if entry != 'source_only':
        from search_operations import diagnosis_prompt as existing_prompt
        return existing_prompt(config, sources, directory, entries, entry, parent)
    return bounded_prompt(parts, config.diagnosis_context.max_input_chars)[0]


def add_manifest_identity(manifest, config, entries):
    from run_metadata import fingerprint_inputs, digest, file_hash, ROOT
    from task_schedule import canonical_tasks, task_hash
    tasks = manifest['tasks']['ordered_ids']
    benchmark = canonical_tasks(ROOT, 42, polyglot=config.evaluation.polyglot)
    overlap = sorted(set(tasks) & set(benchmark))
    manifest['v2_sampling'] = {'schema_version': 1, 'mode': 'uniform_with_replacement',
        'with_replacement': True, 'membership_hash': task_hash(sorted(tasks)), 'order_hash': task_hash(tasks),
        'entry_hashes': {t: io.digest(entries[t]) for t in tasks},
        'task_seed': config.wgm_v2.task_seed, 'selection_seed': config.search.seed,
        'rng_schema': 'python-random-mt19937-state-v3/v1', 'reservation_schema': 'v2_slot_v1',
        'evaluation_id_schema': 'evaluation_plan_v1_with_v2_slot_binding',
        'ledger_root': 'evaluation_ledger', 'benchmark_ordered_ids': benchmark,
        'benchmark_overlap': {'ids': overlap, 'count': len(overlap), 'hash': task_hash(overlap),
                              'task_level_held_out': not overlap},
        'implementation_hashes': {p: file_hash(ROOT / p) for p in (
            'wgm_v2_config.py', 'wgm_v2_selector.py', 'wgm_v2_state.py', 'wgm_v2_search.py')}}
    manifest['wgm_v2_runtime'] = runtime_control(config)
    manifest['fingerprint_inputs'] = fingerprint_inputs(manifest)
    manifest['reproducibility_fingerprint'] = digest(manifest['fingerprint_inputs'])
    return manifest


def run_search(config, *, preflight):
    from execution_policy import record, require_supported
    from run_metadata import ROOT, build_manifest, RunRecorder
    from task_schedule import canonical_tasks, CANONICAL_MEMBERSHIP_HASH, task_hash
    from validated_runs import _entry_map, Sources, executor, refresh_views, _open
    from tree import Node
    import hgm_utils
    config.validate_search()
    check(config.search.algorithm == 'wgm_v2', 'V2_ALGORITHM_REQUIRED')
    requested = config.execution.policy
    config._v2_requested_policy = requested
    if requested == 'origin':
        check(record(config)['policy_source'] != 'default' and config.protocol.effective_name == 'smoke',
              'WGM_V2_ORIGIN_REQUIRES_EXPLICIT_SMOKE')
        warnings.warn('WGM V2 origin request is EXPLORATORY ONLY, not a formal comparable result. '
                      'V2 still uses validated journal/binding semantics; no legacy failure-as-zero path.', stacklevel=2)
        # Runtime readiness contains a cancellation Event; it must remain the
        # same process-local object, not be pickled/deep-copied with its lock.
        config = copy(config)
        config.execution = copy(config.execution)
        config.execution.policy = 'validated'
    require_supported(config)
    check(config.llm.diagnose_backend == config.llm.self_improve_backend == 'codex_exec', 'V2_CODEX_BACKENDS_REQUIRED')
    check(not config.paths.root_cache_dir and not config.paths.initial_baseline_dir, 'V2_ROOT_OBSERVATION_IMPORT_FORBIDDEN')
    config.execution.max_task_evals = 800
    config.execution.max_workers = 1
    directory = config.paths.continue_from or config.paths.output_dir
    check(bool(directory), 'V2_EXPLICIT_OUTPUT_REQUIRED')
    check(not config.paths.continue_from or not config.paths.output_dir or
          Path(config.paths.output_dir).absolute() == Path(directory).absolute(), 'V2_RESUME_DIRECTORY')
    if not config.paths.continue_from:
        from wgm_v2_config import CHECKPOINTS
        check(config.wgm_v2.target_budget != 400 or config.wgm_v2.checkpoints == CHECKPOINTS,
              'V2_CHECKPOINT_MIGRATION_REQUIRED')
    source = resolve_initial_source(config, ROOT)
    from task_schedule import benchmark
    tasks = canonical_tasks(ROOT, config.evaluation.task_order_seed, polyglot=config.evaluation.polyglot)
    check(task_hash(sorted(tasks)) == benchmark(config.evaluation.polyglot)['membership_hash'], 'V2_CANONICAL_POOL_MEMBERSHIP')
    entries = _entry_map(config, tasks)
    proposed = add_manifest_identity(build_manifest(config, directory, 'wgm_v2', tasks, source=source,
        require_clean=True, shuffle=True, seed=config.evaluation.task_order_seed), config, entries)
    resume = bool(config.paths.continue_from)
    # Read-only identity/target checks precede preflight and any model callback.
    if resume:
        saved = io._read_json(Path(directory) / 'run_manifest.json')
        check(saved['reproducibility_fingerprint'] == proposed['reproducibility_fingerprint'], 'V2_RESUME_IDENTITY')
        checkpoint = read_checkpoint(directory)
        check(checkpoint['fingerprint'] == saved['reproducibility_fingerprint'], 'V2_CHECKPOINT_IDENTITY')
        check(saved['wgm_v2_runtime']['protocol'] == config.protocol.effective_name, 'V2_PROTOCOL_CHANGED')
        old, new = saved['wgm_v2_runtime']['target_budget'], pause_target(config)
        check(new == old or old in CHECKPOINTS and new in CHECKPOINTS and CHECKPOINTS.index(new) == CHECKPOINTS.index(old)+1
              and checkpoint['state']['n_task_evals'] == old, 'V2_CONTINUATION_TARGET')
    else:
        check(not Path(directory).exists(), 'RUN_DIRECTORY_EXISTS')
    preflight(config, output_root=directory)
    run = _open(config, directory, proposed, resume)
    with run.lock():
        sources = Sources(run, source)
        recorder = RunRecorder(run.directory, run.manifest, network_readiness=config.network_readiness)
        hgm_utils.nodes.clear()
        hgm_utils.init(config.evaluation.polyglot, str(run.directory), tasks, 0, config.llm.self_improve_llm, config.execution.self_improve_timeout)
        Node('initial')  # No hidden root task evaluations or benchmark cache.
        state = WGMV2State(config, hgm_utils.nodes, defaultdict(set), tasks)
        operations = V2Operations(run, config, sources, entries, recorder, prompt_builder=diagnosis_prompt)
        def make_plan(node, task, draw):
            src = sources.get(node.commit_id)
            return run.plan(entries[task], stage='search', candidate_id=node.commit_id, task_draw_index=draw,
                            policy_fingerprint=src['policy_fingerprint'], source_fingerprint=src['source_fingerprint'])
        engine = WGMV2Search(run, state, Node, make_plan=make_plan,
            execute=executor(config, run, entries, sources, recorder), generate=None, operations=operations,
            refresh_views=lambda: refresh_views(run))
        operations.choose_entry = lambda parent: diagnosis_entry(engine, parent)
        if resume:
            engine.restore()
        else:
            engine.save()
        engine.set_target(pause_target(config))
        try:
            engine.run_to_budget()
        except BaseException as error:
            engine.restore()
            # Fixed codes only; raw errors can contain output, paths or secrets.
            code = getattr(error, 'code', None)
            engine.write_outputs(code if isinstance(code, str) and code.replace('_', '').isalnum() else 'V2_INTERRUPTED')
            recorder.refresh('incomplete')
            raise
        engine.write_outputs()
        recorder.refresh('completed' if engine.count == pause_target(config) else 'incomplete')
    return run.directory


def smoke_coverage(state):
    selections = [r for r in state.history if r['phase'] == 'search']
    return {'initialization_complete': state.initialized, 'founder_count': len(state.founders),
            'exp3_selections': len(state.selector.draws) if state.selector else 0,
            'bai_parent_selections': len(selections), 'iid_draws': len(state.draws),
            'operation_commits': len(state.execution['consumed_operations']),
            'validated_bindings': len(state.execution['consumed_bindings']),
            'post_init_uncovered_reason': None if selections else (
                'insufficient_successful_lineages' if state.initial_attempts >= state.config.wgm_v2.initial_expansion_target
                and len(state.founders) < 2 else 'pause_before_post_initialization')}


def recorded_wall_time(recorder):
    import time
    started, previous = getattr(recorder, 'started', None), getattr(recorder, 'previous_total', None)
    return previous + time.monotonic() - started if type(started) in (int, float) and type(previous) in (int, float) else None
