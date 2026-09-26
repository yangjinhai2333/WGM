"""Explicit checkpoint evaluations. Canonical inputs are read-only; execution uses N1/ledger."""
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import tempfile

import evaluation_ledger as io
from evaluation_plan import EvaluationPlan, check, durable_json, execution_identity
from evaluation_scheduler import ExecutionRun, SchedulingBlocked, read_binding
from run_metadata import digest, fingerprint_inputs
from search_operation_journal import SearchOperationJournal, durable_record
from task_schedule import schedule_identity
from wgm_v2_config import CHECKPOINTS, SPEC
from wgm_v2_selector import bai, interval, restore_rng, rng_snapshot


def derived_seed(checkpoint_digest, b, task_seed):
    raw = json.dumps([SPEC, 'derived-task', checkpoint_digest, b, task_seed],
                     ensure_ascii=True, separators=(',', ':')).encode()
    return int.from_bytes(hashlib.sha256(raw).digest(), 'big')


def snapshot_rows(state, counts=None):
    delta = state['experiment']['parameters']['delta']
    rows = []
    for node in state['nodes']:
        cid = node['commit_id']
        n, s = (counts[cid] if counts is not None else
                (len(node['utility_measures']), sum(node['utility_measures'])))
        rows.append({'candidate_id': cid, 'node_id': node['id'],
            'lineage_id': state['branches'][str(node['id'])],
            'admission_index': state['admission'][cid],
            **interval(state['admission'][cid], n, s, delta)})
    return rows


def snapshot_selection(state):
    rows, ranked = snapshot_rows(state), []
    nodes = {r['id']: r for r in state['nodes']}
    for lineage in sorted(int(c) for c in state['founders']):
        top = bai([r for r in rows if r['lineage_id'] == lineage],
                  state['experiment']['parameters']['epsilon'])['leader']
        node, chain = nodes[top['node_id']], []
        while node is not None:
            chain.append(node['commit_id'])
            node = nodes.get(node['parent_id'])
        ranked.append({**top, 'parent_chain': list(reversed(chain)),
                       'selection_basis': 'maximum_search_lower_bound',
                       'tie_break': 'candidate_utf8_bytes', 'certificate': None})
    ranked.sort(key=lambda r: (-r['lower'], r['lineage_id']))
    return {'primary': ranked[0] if ranked else None, 'secondary': ranked[1] if len(ranked) > 1 else None,
            'secondary_reason': None if len(ranked) > 1 else 'fewer_than_two_eligible_lineages'}


class Checkpoint:
    def __init__(self, directory, b):
        self.directory = io._checked_path(directory)
        check(type(b) is int and b in CHECKPOINTS, 'DERIVED_FORMAL_CHECKPOINT_REQUIRED')
        self.b = b
        self.manifest = io._read_json(self.directory / 'run_manifest.json')
        ExecutionRun(self.directory, self.manifest)  # Read-only identity/schema verification.
        self.row = io._read_json(self.directory / 'checkpoints' / f'{b:06d}.json')
        self.state = self.row['state']
        self.digest = self.row['state_digest']
        check(self.row['schema_version'] == 1 and self.row['domain'] == 'canonical' and self.row['b'] == b
              and self.row['run_fingerprint'] == self.manifest['reproducibility_fingerprint']
              and self.digest == digest(self.state) and self.state['n_task_evals'] == b
              and self.state['experiment'] == self.manifest['experiment']
              and self.state['experiment']['algorithm'] == 'wgm_v2', 'DERIVED_CHECKPOINT_IDENTITY')
        self.selection = snapshot_selection(self.state)
        saved_selection = io._read_json(self.directory / 'checkpoints' / f'{b:06d}.selection.json')
        check(saved_selection['canonical_checkpoint_digest'] == self.digest and
              all(saved_selection[k] == v for k, v in self.selection.items()), 'DERIVED_SELECTION_CHANGED')
        self.pool = self.manifest['tasks']['ordered_ids']
        self.final_tasks = self.manifest['v2_sampling']['benchmark_ordered_ids']
        from task_schedule import canonical_tasks, manifest_polyglot, pool_size
        from run_metadata import ROOT
        polyglot = manifest_polyglot(self.manifest)
        check(self.final_tasks == canonical_tasks(ROOT, 42, polyglot=polyglot)
              and len(self.final_tasks) == pool_size(polyglot),
              'DERIVED_FINAL60_SCHEDULE')
        self.output = self.directory / 'derived' / f'checkpoint-{b:06d}'

    def unchanged(self):
        check(io._read_json(self.directory / 'checkpoints' / f'{self.b:06d}.json') == self.row,
              'DERIVED_CANONICAL_CHANGED')


class CheckpointSources:
    """Replay schema-v1 ancestors once into derived-owned caches, never canonical caches."""
    def __init__(self, checkpoint, directory):
        from run_metadata import source_snapshot
        self.checkpoint, self.directory = checkpoint, Path(directory)
        self.root = checkpoint.directory / 'initial/src'
        check(source_snapshot(self.root) == checkpoint.manifest['policy']['initial_source_snapshot'],
              'DERIVED_ROOT_SOURCE_CHANGED')
        self.journal = SearchOperationJournal(checkpoint.directory, checkpoint.manifest['run_id'],
                                             checkpoint.manifest['reproducibility_fingerprint'])
        self.plans = {p.identity['candidate_id']: p for p in self.journal.plans()}
        self.allowed = {r['commit_id'] for r in checkpoint.state['nodes']}

    def get(self, candidate, *, read_only=False):
        from candidate_provenance import read_chain, replay, source_digest
        from candidate_workspace import _copy_source, _git
        from downstream_solver import load_agent_policy
        from run_metadata import source_snapshot
        check(candidate in self.allowed, 'DERIVED_CANDIDATE_NOT_IN_CHECKPOINT')
        check(source_snapshot(self.root) == self.checkpoint.manifest['policy']['initial_source_snapshot'],
              'DERIVED_ROOT_SOURCE_CHANGED')
        chain = read_chain(self.checkpoint.directory, candidate, require_versioned=True)
        for item in chain:
            cid = item.provenance.candidate_id
            check(cid in self.plans, 'DERIVED_CANDIDATE_OPERATION_MISSING')
            plan = self.plans[cid]
            state, _ = self.journal.inspect(plan)
            terminal = state['terminal']
            check(terminal is not None and terminal['outcome'] == 'candidate' and
                  self.checkpoint.state['execution']['consumed_operations'].get(plan.operation_id) == {
                    'operation_terminal_digest': terminal['terminal_digest'],
                    'candidate_digest': terminal['candidate_digest']}, 'DERIVED_OPERATION_BINDING')
            artifact = self.journal.read_artifact(plan, terminal['artifact'])
            metadata = io._read_json(self.checkpoint.directory / cid / 'metadata.json')
            check({k:v for k,v in metadata.items() if k != 'overall_performance'} ==
                  {k:v for k,v in artifact['metadata'].items() if k != 'overall_performance'} and
                  hashlib.sha256(item.patch_bytes).hexdigest() == artifact['patch_digest'] and
                  digest(item.provenance.to_dict()) == terminal['candidate_digest'], 'DERIVED_PROVENANCE_CHANGED')
        path = self.root
        if chain:
            folder = self.directory / 'evaluation_sources'
            path = folder / candidate
            check(not read_only or path.is_dir(), 'DERIVED_SOURCE_MISSING')
            if not read_only:
                folder.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                with tempfile.TemporaryDirectory(prefix='.tmp-', dir=folder) as temporary:
                    staged = Path(temporary) / 'agent'
                    _copy_source(self.root, staged, False, preserve_source=True)
                    _git(staged, 'init', '-q', '--template=')
                    replay(staged, chain)
                    staged.rename(path)
                    io._sync_dir(folder)
            check(source_digest(path) == chain[-1].provenance.candidate_source_sha256, 'DERIVED_SOURCE_CHANGED')
        policy = load_agent_policy(path)
        return {'path': path, 'policy': policy, 'policy_fingerprint': hashlib.sha256(policy.encode()).hexdigest(),
                'source_fingerprint': source_digest(path),
                'chain_digest': digest([r.provenance.to_dict() for r in chain])}


def make_manifest(checkpoint, config, domain, entries):
    check(domain in ('bai', 'primary', 'secondary'), 'DERIVED_DOMAIN')
    selected = checkpoint.selection.get(domain) if domain != 'bai' else None
    check(domain == 'bai' or selected is not None, 'DERIVED_SECONDARY_UNAVAILABLE')
    tasks = checkpoint.pool if domain == 'bai' else checkpoint.final_tasks
    check(all(digest(entries[t]) == checkpoint.manifest['v2_sampling']['entry_hashes'][t] for t in tasks),
          'DERIVED_TASK_INPUT_CHANGED')
    row = deepcopy(checkpoint.manifest)
    params = checkpoint.state['experiment']['parameters']
    check(domain != 'bai' or params['checkpoint_bai_fraction'] == .2, 'DERIVED_FORMAL_BAI_FRACTION_REQUIRED')
    row['v2_derived'] = {'schema_version': 1, 'domain': domain, 'checkpoint': checkpoint.b,
        'canonical_run_id': checkpoint.manifest['run_id'], 'canonical_run_fingerprint': checkpoint.manifest['reproducibility_fingerprint'],
        'canonical_checkpoint_digest': checkpoint.digest, 'selection': selected,
        'budget': math.floor(params['checkpoint_bai_fraction'] * checkpoint.b) if domain == 'bai' else len(tasks),
        'task_seed': derived_seed(checkpoint.digest, checkpoint.b, params['task_seed']) if domain == 'bai' else None,
        'entry_hashes': {t: digest(entries[t]) for t in tasks},
        'sampling': 'uniform_with_replacement' if domain == 'bai' else 'fixed_final60_once'}
    row['run_id'] = 'derived-' + digest(row['v2_derived'])
    row['run_type'] = 'wgm_v2'  # Preserve the existing V2 configuration/scheduler identity.
    from evaluation_batch import runtime_workers
    row['evaluation_parallelism'] = runtime_workers(config, 'wgm_v2', len(tasks), domain=domain)
    row['tasks'].update(ordered_ids=list(tasks), count=len(tasks), sha256=digest(tasks),
                        **schedule_identity(tasks, 42 if domain != 'bai' else row['tasks']['task_order_seed']))
    row['evaluation_execution'] = execution_identity(config, row)
    row['fingerprint_inputs'] = fingerprint_inputs(row)
    row['reproducibility_fingerprint'] = digest(row['fingerprint_inputs'])
    return row


class DerivedEvaluation:
    """One serial derived domain, durable pending plan and binding consumption; no search feedback."""
    def __init__(self, checkpoint, run, entries, sources, execute, *, fault=lambda point: None):
        self.checkpoint, self.run, self.entries, self.sources, self.execute = checkpoint, run, entries, sources, execute
        self.fault = fault
        self.identity = run.manifest['v2_derived']
        self.domain = self.identity['domain']
        self.path = run.directory / 'derived_state.json'

    def initial(self):
        return {'schema_version': 1, 'identity': digest(self.identity), 'records': [], 'pending': None,
                'rng': rng_snapshot(random.Random(self.identity['task_seed'])) if self.domain == 'bai' else None,
                'counts': {r['candidate_id']: [r['n'], r['successes']] for r in snapshot_rows(self.checkpoint.state)}}

    def decision(self, state):
        if self.domain == 'bai':
            choice = bai(snapshot_rows(self.checkpoint.state, state['counts']),
                         self.checkpoint.state['experiment']['parameters']['epsilon'])
            reason = choice['stop_reason'] or ('budget_exhausted' if len(state['records']) >= self.identity['budget'] else None)
            return choice, reason
        return None, 'final60_completed' if len(state['records']) == self.identity['budget'] else None

    def next_plan(self, state):
        choice, reason = self.decision(state)
        check(reason is None, 'DERIVED_ALREADY_COMPLETE')
        ordinal = len(state['records'])
        if self.domain == 'bai':
            rng = restore_rng(state['rng'])
            task = self.checkpoint.pool[rng.randrange(len(self.checkpoint.pool))]
            state['rng'] = rng_snapshot(rng)
            cid = choice['evaluate']['candidate_id']
        else:
            task = self.checkpoint.final_tasks[ordinal]
            cid = self.identity['selection']['candidate_id']
        source = self.sources.get(cid)
        return self.run.plan(self.entries[task], stage='checkpoint_bai' if self.domain == 'bai' else 'final_benchmark',
            candidate_id=cid, policy_fingerprint=source['policy_fingerprint'], source_fingerprint=source['source_fingerprint'],
            ordinal=ordinal, task_draw_index=ordinal if self.domain == 'bai' else None,
            selection_digest=digest(self.identity))

    def consume(self, state, plan, result):
        check(result.outcome == 'success' and type(result.resolved) is bool, 'DERIVED_REQUIRES_TRUSTED_RESULT')
        check(result == read_binding(self.run.ledger, plan), 'DERIVED_BINDING_CHANGED')
        if self.domain == 'bai':
            counts = state['counts'][plan.identity.candidate_id]
            counts[0] += 1
            counts[1] += int(result.resolved)
        state['records'].append({'plan': asdict(plan), 'binding_digest': result.binding_digest,
                                 'resolved': result.resolved})
        state['pending'] = None

    def save(self, state):
        durable_json(self.path, {'state': state, 'sha256': digest(state)}, replace_existing=True)

    def load(self):
        state = self.initial()
        if not self.path.exists():
            check(not self.run.plans.read(), 'DERIVED_STATE_MISSING')
            self.save(state)
            return state
        envelope = io._read_json(self.path)
        saved = envelope['state']
        check(envelope['sha256'] == digest(saved) and saved['identity'] == state['identity'], 'DERIVED_STATE_DIGEST')
        # Replay only local deterministic decisions and already-bound results. No callback/RNG reuse.
        for record in saved['records']:
            plan = self.next_plan(state)
            check(record['plan'] == asdict(plan), 'DERIVED_PLAN_CHANGED')
            result = read_binding(self.run.ledger, plan)
            self.consume(state, plan, result)
            check(state['records'][-1] == record, 'DERIVED_BINDING_CHANGED')
        if saved['pending'] is not None:
            state['pending'] = asdict(self.next_plan(state))
        check(state == saved, 'DERIVED_RNG_OR_STATE_CHANGED')
        plans = self.run.plans.read()
        expected = ([asdict(p) for p in self.final_plans()] if self.domain != 'bai' else
                    [r['plan'] for r in state['records']] + ([state['pending']] if state['pending'] else []))
        check([asdict(p) for p in plans] == expected[:len(plans)], 'DERIVED_ORPHAN_PLAN')
        return state

    def final_plans(self):
        """Pure fixed schedule; no BAI/task RNG, records or completion-order inputs."""
        check(self.domain in ('primary', 'secondary', 'bai_final'), 'DERIVED_FINAL_REQUIRED')
        cid = self.identity['selection']['candidate_id']
        source = self.sources.get(cid)  # Materialize the immutable source before workers.
        return [self.run.plan(self.entries[task], stage='final_benchmark', candidate_id=cid,
                    policy_fingerprint=source['policy_fingerprint'], source_fingerprint=source['source_fingerprint'],
                    ordinal=i, selection_digest=digest(self.identity)) for i, task in enumerate(self.checkpoint.final_tasks)]

    def run_to_completion(self, *, locked=False, workers=1, cancel_event=None):
        with nullcontext() if locked else self.run.lock():
            state = self.load()
            consumed = {r['plan']['identity']['evaluation_id']: r['binding_digest'] for r in state['records']}
            self.run.precheck(consumed)
            try:
                if self.domain != 'bai':
                    from evaluation_batch import execute_batch
                    self.checkpoint.unchanged()
                    def consume(plan, result):
                        if plan.identity.ordinal < len(state['records']):
                            check(state['records'][plan.identity.ordinal]['binding_digest'] == result.binding_digest,
                                  'DERIVED_BINDING_CHANGED')
                            return
                        check(plan.identity.ordinal == len(state['records']), 'DERIVED_CONSUMPTION_ORDER')
                        self.fault('derived.after_binding')
                        self.consume(state, plan, result)
                        self.save(state)
                        self.fault('derived.after_consumption')
                    execute_batch(self.run, self.final_plans(), self.execute, workers=workers,
                                  consume=consume, cancel_event=cancel_event)
                    return self.summary(state)
                while state['pending'] is not None or self.decision(state)[1] is None:
                    if state['pending'] is None:
                        state['pending'] = asdict(self.next_plan(state))
                        self.save(state)  # Plan + post-draw RNG durable before ledger/model permission.
                        self.fault('derived.after_plan')
                    plan = EvaluationPlan.from_dict(state['pending'])
                    self.checkpoint.unchanged()
                    source = self.sources.get(plan.identity.candidate_id)
                    check(all(source[k] == getattr(plan.identity, k) for k in ('source_fingerprint', 'policy_fingerprint')),
                          'DERIVED_SOURCE_CHANGED')
                    result = self.run.execute(plan, self.execute)
                    self.fault('derived.after_binding')
                    self.consume(state, plan, result)
                    self.save(state)
                    self.fault('derived.after_consumption')
                return self.summary(state)
            finally:
                # Read the last committed state; failed local writes must not invent consumption.
                committed = self.load()
                durable_json(self.run.directory / 'summary.json', self.summary(committed), replace_existing=True)
                self.checkpoint.unchanged()

    def summary(self, state):
        choice, reason = self.decision(state)
        done = reason is not None and state['pending'] is None
        result = {'schema_version': 1, 'domain': self.domain, 'checkpoint_digest': self.checkpoint.digest,
            'completed': len(state['records']), 'requested': self.identity['budget'],
            'pending': 0 if done else self.identity['budget'] - len(state['records']),
            'status': 'completed' if done else 'incomplete', 'stop_reason': reason if done else None,
            'canonical_feedback': False, 'task_level_held_out': False, 'D_design': None}
        if self.domain == 'bai':
            result.update(selection=choice['leader'], challenger=choice['challenger'],
                certificate='conditional_epsilon' if done and reason == 'epsilon_satisfied' else None,
                confidence_scope='one_IID_stationary_trajectory_not_joint_checkpoints',
                draws=[{'task_id': r['plan']['identity']['task_id'], 'evaluation_id': r['plan']['identity']['evaluation_id'],
                        'candidate_id': r['plan']['identity']['candidate_id']} for r in state['records']])
            if done:
                path = self.run.directory / 'selection.json'
                selection = {**result, 'canonical_selection_overwritten': False}
                if path.exists():
                    check(io._read_json(path) == selection, 'DERIVED_SELECTION_CONFLICT')
                else:
                    durable_record(path, selection)
        else:
            outcomes = {r['plan']['identity']['task_id']: r['resolved'] for r in state['records']}
            result.update(final_metrics(self.identity['selection'], self.checkpoint.final_tasks, outcomes))
        from evaluation_batch import batch_reports
        result.update(evaluation_parallelism=self.run.manifest.get('evaluation_parallelism'),
                      evaluation_batches=batch_reports(self.run.directory))
        return result


def final_metrics(selection, tasks, outcomes):
    check(set(outcomes) <= set(tasks) and all(type(v) is bool for v in outcomes.values()), 'FINAL_OUTCOME_SCHEMA')
    from task_schedule import BENCHMARKS, task_hash
    # Accuracy is reported only for a benchmark's complete canonical pool (SWE 60 / Polyglot 59).
    canonical = task_hash(sorted(tasks)) in {b['membership_hash'] for b in BENCHMARKS.values()}
    complete = canonical and len(outcomes) == len(tasks)
    solved = [t for t in tasks if outcomes.get(t) is True]
    return {'candidate_id': selection['candidate_id'], 'lineage_id': selection['lineage_id'],
        'completed': len(outcomes), 'pending': len(tasks)-len(outcomes), 'complete': complete,
        'outcomes': outcomes, 'resolved': len(solved), 'denominator': len(tasks),
        'accuracy': len(solved)/len(tasks) if complete else None, 'solved_set': solved if complete else None,
        'observed_solved_set': solved}


def diversity(primary, secondary):
    keys = ('exclusive_primary_solves', 'exclusive_secondary_solves', 'exclusive_primary_count',
        'exclusive_secondary_count', 'union_resolved', 'union_coverage', 'intersection', 'intersection_count',
        'jaccard_similarity', 'empty_union', 'hamming_distance', 'normalized_hamming_distance')
    result = dict.fromkeys(keys)
    if not primary or not secondary or not primary.get('complete') or not secondary.get('complete'):
        return {**result, 'status': 'incomplete', 'reason': 'requires_both_complete_final60'}
    check(primary['lineage_id'] != secondary['lineage_id'], 'FINAL_LINEAGES_NOT_DISTINCT')
    a, b = set(primary['solved_set']), set(secondary['solved_set'])
    total = primary['denominator']
    union, intersection = a | b, a & b
    return {'status': 'complete', 'exclusive_primary_solves': sorted(a-b), 'exclusive_secondary_solves': sorted(b-a),
        'exclusive_primary_count': len(a-b), 'exclusive_secondary_count': len(b-a),
        'union_resolved': len(union), 'union_coverage': len(union)/total, 'union_is_single_agent_accuracy': False,
        'intersection': sorted(intersection), 'intersection_count': len(intersection),
        'jaccard_similarity': len(intersection)/len(union) if union else 1., 'empty_union': not union,
        'hamming_distance': len(a ^ b), 'normalized_hamming_distance': len(a ^ b)/total}


def cache_identity(manifest):
    from execution_policy import semantics
    execution = manifest.get('evaluation_execution', {})
    required = ('model_fingerprint', 'evaluator_fingerprint', 'source_fingerprint', 'policy_fingerprint')
    check(all(execution.get(k) for k in required), 'ROOT_CACHE_EXECUTOR_IDENTITY_MISSING')
    check(bool(manifest['codex']['configured_model']) and bool(manifest['codex']['configured_reasoning_effort']),
          'ROOT_CACHE_PINNED_MODEL_REQUIRED')
    return {'commit': manifest['repository']['commit'], 'policy': semantics(manifest.get('execution_policy')),
        'model': manifest['codex']['configured_model'], 'reasoning': manifest['codex']['configured_reasoning_effort'],
        'cli_version': manifest['codex']['codex_cli_version'],
        'tasks': {k: manifest['tasks'][k] for k in ('ordered_ids', 'task_membership_hash', 'task_order_hash', 'task_order_seed')},
        'source': manifest['policy'], 'dataset': manifest['dataset'], 'evaluator': manifest.get('evaluator'),
        'executor': {k: execution[k] for k in required}}


def root_cache_result(directory, target, selection, tasks):
    """Reuse either a verified import or a complete validated root/final60 ledger, read-only."""
    from root_cache import verify_root_cache
    check(selection['candidate_id'] == 'initial', 'ROOT_CACHE_ONLY_INITIAL')
    directory = Path(directory)
    if (directory / 'evaluation_ledger/run.json').exists():
        baseline = io._read_json(directory / 'run_manifest.json')
        cache = ExecutionRun(directory, baseline)
        check(cache_identity(baseline) == cache_identity(target), 'ROOT_CACHE_IDENTITY_MISMATCH')
        plans = cache.plans.read()
        check(len(plans) == len(tasks) and [p.identity.task_id for p in plans] == tasks and all(
            p.identity.candidate_id == 'initial' and p.identity.stage in ('root', 'final_benchmark') for p in plans),
            'ROOT_CACHE_REQUIRES_FINAL60_NOT_SEARCH')
        outcomes, evidence = {}, []
        for plan in plans:
            for field in ('source_fingerprint', 'policy_fingerprint'):
                check(getattr(plan.identity, field) == target['evaluation_execution'][field], 'ROOT_CACHE_SOURCE_MISMATCH')
            check(plan.identity.task_input_fingerprint == target['v2_derived']['entry_hashes'][plan.identity.task_id],
                  'ROOT_CACHE_TASK_INPUT_MISMATCH')
            result = read_binding(cache.ledger, plan)
            check(result.outcome == 'success', 'ROOT_CACHE_INCOMPLETE')
            outcomes[plan.identity.task_id] = result.resolved
            evidence.append({'task_id': plan.identity.task_id, 'evaluation_id': plan.identity.evaluation_id,
                             'binding_digest': result.binding_digest})
        content_digest = digest(evidence)
    else:
        cached = verify_root_cache(directory, target=target)
        baseline = io._read_json(directory / 'evidence/run_manifest.json')
        check(cache_identity(baseline) == cache_identity(target), 'ROOT_CACHE_IDENTITY_MISMATCH')
        check(cached['compatibility']['ordered_task_ids'] == tasks, 'ROOT_CACHE_SCHEDULE_MISMATCH')
        outcomes = {r['task_id']: r['result'] == 'resolved' for r in cached['per_task_results']}
        check(len(outcomes) == len(tasks) and all(r['result'] in ('resolved', 'unresolved') for r in cached['per_task_results']),
              'ROOT_CACHE_INCOMPLETE')
        content_digest, evidence = cached['content_digest'], cached['per_task_results']
    return {**final_metrics(selection, tasks, outcomes), 'cache_reuse': {
        'root_cache_digest': content_digest, 'baseline_fingerprint': baseline['reproducibility_fingerprint'],
        'execution_policy': baseline['execution_policy'], 'physical_calls_in_this_domain': 0,
        'evidence_digest': digest(evidence)},
        'reused_evaluations': [{'task_id': t, 'evaluation_id': 'reuse-' + digest({
            'derived_identity': target['v2_derived'], 'task_id': t, 'cache_digest': content_digest})} for t in tasks],
        'status': 'completed', 'canonical_feedback': False}


def read_root_reuse(path, checkpoint_digest, fingerprint=None):
    value = io._read_json(path)
    check(value.get('reuse_digest') == digest({k:v for k,v in value.items() if k != 'reuse_digest'})
          and value.get('checkpoint_digest') == checkpoint_digest
          and (fingerprint is None or value.get('derived_fingerprint') == fingerprint), 'ROOT_CACHE_REUSE_CHANGED')
    return value


def accounting(directory, *, evaluation_ids=None, operation_ids=None):
    """Physical confirmations and recorded durations, never inferred missing terminals."""
    directory = Path(directory)
    manifest = io._read_json(directory / 'run_manifest.json')
    run = ExecutionRun(directory, manifest)
    model, evaluator = set(), set()
    markers = {'model': 0, 'evaluation': 0}
    ambiguous = 0
    times = {'downstream_model': [], 'downstream_evaluator': [], 'diagnosis': [], 'generation': []}
    for snapshot in run.ledger.evaluations():
        if evaluation_ids is not None and snapshot.identity.evaluation_id not in evaluation_ids:
            continue
        ambiguous += int('AMBIGUOUS' in io.classify(run.ledger.inspect(snapshot.identity.evaluation_id)).value)
        for event in snapshot.events:
            if event.kind in ('model_started', 'evaluation_started'):
                markers[event.kind.removesuffix('_started')] += 1
            data = io._decode(event.data)
            if event.kind == 'model_start_confirmed':
                model.add(io.invocation_id(snapshot.identity, event.attempt_id, 'model'))
            elif event.kind == 'evaluation_start_confirmed':
                evaluator.add(io.invocation_id(snapshot.identity, event.attempt_id, 'evaluator'))
            elif event.kind in ('model_terminal', 'evaluation_terminal'):
                key = 'downstream_model' if event.kind == 'model_terminal' else 'downstream_evaluator'
                times[key].append(data.get('evidence', {}).get('duration_seconds'))
    op_calls = {'diagnosis': set(), 'generation': set()}
    if (directory / 'search_operations/head.json').exists():
        from search_operation_journal import call_id
        journal = SearchOperationJournal(directory, manifest['run_id'], manifest['reproducibility_fingerprint'])
        for plan in journal.plans():
            if operation_ids is not None and plan.operation_id not in operation_ids:
                continue
            state, _ = journal.inspect(plan)
            for stage in op_calls:
                for attempt in state[stage]:
                    if attempt.get('started'):
                        op_calls[stage].add(call_id(plan, stage, attempt['attempt']))
        for stage, ids in op_calls.items():
            for cid in ids:
                path = directory / 'codex_calls' / (cid + '.json')
                times[stage].append(io._read_json(path).get('duration_seconds') if path.exists() else None)
            model.update(ids)
    def total(values, expected=None):
        if expected is not None and len(values) != expected:
            return None
        return sum(values) if all(type(t) in (int, float) and math.isfinite(t) and t >= 0 for t in values) else None
    # Missing terminal duration is unknown, never a fabricated zero.
    model_time = total(times['downstream_model'], markers['model'])
    eval_time = total(times['downstream_evaluator'], markers['evaluation'])
    return {'physical_model_calls': len(model), 'physical_evaluator_calls': len(evaluator),
        'total_physical_calls': len(model)+len(evaluator),
        'evaluation_model_markers': markers['model'], 'evaluator_markers': markers['evaluation'],
        'ambiguous_evaluations': ambiguous,
        'diagnosis_calls': len(op_calls['diagnosis']), 'generation_calls': len(op_calls['generation']),
        'diagnosis_time_seconds': total(times['diagnosis']), 'generation_time_seconds': total(times['generation']),
        'downstream_evaluation_time_seconds': None if model_time is None or eval_time is None else model_time+eval_time,
        'time_definition': 'recorded_process_durations; downstream=model+grader; missing_evidence=null',
        'call_definition': 'local_process_start_confirmations_not_remote_API_requests'}


def report(checkpoint):
    domains, counts = {}, {}
    for domain in ('bai', 'primary', 'secondary'):
        folder = checkpoint.output / domain
        summary_path = folder / 'summary.json'
        domains[domain] = (read_root_reuse(folder / 'root_reuse.json', checkpoint.digest)
                           if (folder / 'root_reuse.json').exists() else
                           io._read_json(summary_path) if summary_path.exists() else None)
        if domains[domain] is not None:
            check(domains[domain]['checkpoint_digest'] == checkpoint.digest, 'DERIVED_REPORT_IDENTITY')
        if (folder / 'run_manifest.json').exists():
            counts[domain] = accounting(folder)
            summary_file = folder / 'run_summary.json'
            counts[domain]['wall_time_seconds'] = (io._read_json(summary_file).get('timing', {}).get('whole_run_total')
                                                  if summary_file.exists() else None)
    state = checkpoint.state
    canonical = accounting(checkpoint.directory, evaluation_ids=set(state['execution']['consumed_bindings']),
                           operation_ids=set(state['execution']['consumed_operations']))
    canonical.update(valid_search_observations=state['n_task_evals'], canonical_state_digest=checkpoint.digest,
                     canonical_search_wall_time_seconds=checkpoint.row.get('search_wall_time_seconds'),
                     task_draws=len(state['draws']), repeat_draws=state['summary']['repeat_draws'])
    for domain in ('bai', 'primary', 'secondary'):
        if domains[domain] is not None:
            domains[domain]['accounting'] = counts.get(domain, {'total_physical_calls': 0, 'cache_reuse': True})
    out = {'schema_version': 1, 'checkpoint': checkpoint.b, 'canonical_search': canonical,
        'checkpoint_bai': domains['bai'], 'primary_final': domains['primary'], 'secondary_final': domains['secondary'],
        'selection': checkpoint.selection, 'diversity': diversity(domains['primary'], domains['secondary']),
        'time_seconds': {'canonical_search_wall': canonical['canonical_search_wall_time_seconds'],
            'diagnosis': canonical['diagnosis_time_seconds'], 'generation': canonical['generation_time_seconds'],
            'downstream_evaluation': canonical['downstream_evaluation_time_seconds'],
            **{name: counts.get(domain, {}).get('downstream_evaluation_time_seconds') for name,domain in
               (('checkpoint_bai', 'bai'), ('primary_final', 'primary'), ('secondary_final', 'secondary'))}},
        'total_physical_call_counts': canonical['total_physical_calls'] + sum(c['total_physical_calls'] for c in counts.values()),
        'accounting_scope': 'canonical_checkpoint_prefix_plus_this_checkpoint_derived_domains; cache_original_cost_excluded',
        'protocol': checkpoint.row['runtime_control']['protocol'],
        'formal_comparison_eligible': checkpoint.row['runtime_control']['protocol'] == 'formal' and
            checkpoint.manifest['experiment']['requested_execution_policy'] == 'validated',
        'task_level_held_out': False, 'D_design': None, 'D_design_reason': 'design_feature_schema_not_frozen',
        'secondary_reason': checkpoint.selection['secondary_reason'], 'production_ready': False,
        'retained_blocker': 'TRANSIENT_NETWORK_REPLACEMENT_NOT_IMPLEMENTED'}
    checkpoint.output.mkdir(parents=True, exist_ok=True)
    durable_json(checkpoint.output / 'report.json', out, replace_existing=True)
    checkpoint.unchanged()
    return out


def run_derived(config, directory, b, mode, *, roles='primary-only', root_cache=None, preflight,
                retry_preparation=None):
    """CLI boundary; no external work until immutable/config/source identity checks pass."""
    from run_metadata import ROOT, RunRecorder, effective_configuration, repository_snapshot, check_repository
    from validated_runs import _entry_map, executor
    from execution_policy import require_supported
    config.validate_search()
    require_supported(config)
    check(config.search.algorithm == 'wgm_v2', 'DERIVED_V2_REQUIRED')
    check(retry_preparation is None or mode == 'final' and roles in ('primary-only', 'secondary-only'),
          'PREPARATION_RETRY_SINGLE_ROLE_REQUIRED')
    checkpoint = Checkpoint(directory, b)
    repository = repository_snapshot(ROOT)
    check_repository(repository)
    old_commit = checkpoint.manifest['repository']['commit']
    check(repository['commit'] == old_commit, 'DERIVED_COMMIT_CHANGED')
    proposed = deepcopy(checkpoint.manifest)
    proposed['configuration']['effective'] = effective_configuration(config, 'wgm_v2')
    check(fingerprint_inputs(proposed)['configuration'] == checkpoint.manifest['fingerprint_inputs']['configuration'],
          'DERIVED_CONFIGURATION_CHANGED')
    entries = _entry_map(config, checkpoint.pool)
    domains = ['bai'] if mode == 'bai' else {'primary-only': ['primary'], 'both': ['primary', 'secondary'],
        'primary-secondary': ['primary', 'secondary'], 'secondary-only': ['secondary']}[roles]
    for domain in domains:
        check(domain == 'bai' or checkpoint.selection[domain] is not None, 'DERIVED_SECONDARY_UNAVAILABLE')
    # Each independent domain owns its existing scheduler lease. Canonical has no writer here.
    for domain in domains:
        folder = checkpoint.output / domain
        manifest = make_manifest(checkpoint, config, domain, entries)
        folder.parent.mkdir(parents=True, exist_ok=True)
        reuse_path = folder / 'root_reuse.json'
        if reuse_path.exists() or (root_cache is not None and domain != 'bai' and
                                    checkpoint.selection[domain]['candidate_id'] == 'initial'):
            source = CheckpointSources(checkpoint, folder).get('initial')
            check(all(source[k] == manifest['evaluation_execution'][k] for k in
                      ('source_fingerprint', 'policy_fingerprint')), 'ROOT_CACHE_SOURCE_MISMATCH')
        if root_cache is not None and domain != 'bai' and checkpoint.selection[domain]['candidate_id'] == 'initial':
            result = root_cache_result(root_cache, manifest, checkpoint.selection[domain], checkpoint.final_tasks)
            check(not (folder / 'evaluation_ledger').exists(), 'ROOT_CACHE_CANNOT_REPLACE_EXISTING_EVALUATIONS')
            folder.mkdir(exist_ok=True)
            with io._exclusive(folder / 'root_reuse.lock'):
                value = {**result, 'domain': domain, 'checkpoint_digest': checkpoint.digest,
                         'derived_fingerprint': manifest['reproducibility_fingerprint']}
                value['reuse_digest'] = digest(value)
                if reuse_path.exists():
                    check(io._read_json(reuse_path) == value, 'ROOT_CACHE_REUSE_CHANGED')
                else:
                    durable_record(reuse_path, value)
                durable_json(folder / 'summary.json', value, replace_existing=True)
            continue
        if reuse_path.exists():
            read_root_reuse(reuse_path, checkpoint.digest, manifest['reproducibility_fingerprint'])
            continue  # Frozen verified evidence; never create fresh invocations over reused final tasks.
        if folder.exists():
            saved = io._read_json(folder / 'run_manifest.json')
            check(saved['reproducibility_fingerprint'] == manifest['reproducibility_fingerprint'], 'DERIVED_RESUME_IDENTITY')
            run = ExecutionRun(folder, saved)
        else:
            run = ExecutionRun.create(folder, manifest)
        with run.lock():
            if domain != 'bai':
                from preparation_recovery import check_implementation
                check_implementation(run, ROOT, old_commit, repository, retry_preparation)
            run.precheck()
            sources = CheckpointSources(checkpoint, folder)
            candidates = [r['commit_id'] for r in checkpoint.state['nodes']] if domain == 'bai' else [checkpoint.selection[domain]['candidate_id']]
            for candidate in candidates:
                sources.get(candidate)
            # Preflight is read-only and skipped entirely for already completed domains.
            existing = io._read_json(folder / 'summary.json') if (folder / 'summary.json').exists() else None
            if not existing or existing['status'] != 'completed':
                preflight(config, output_root=folder)
            recorder = RunRecorder(folder, run.manifest, network_readiness=config.network_readiness)
            import threading
            cancel = threading.Event()
            execute = executor(config, run, entries, sources, recorder, cancel_event=cancel)
            engine = DerivedEvaluation(checkpoint, run, entries, sources, execute)
            try:
                engine.run_to_completion(locked=True, workers=config.execution.evaluation_workers, cancel_event=cancel)
            except BaseException:
                recorder.refresh('incomplete')
                report(checkpoint)
                raise
            recorder.refresh('completed')
    return report(checkpoint)
