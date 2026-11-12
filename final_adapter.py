"""External comparison selection and final60. Search inputs are read-only."""
from copy import deepcopy
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
import threading

import evaluation_ledger as io
from evaluation_plan import check, durable_json, execution_identity
from evaluation_scheduler import ExecutionRun, read_binding
from run_metadata import ROOT, digest, fingerprint_inputs
from task_schedule import canonical_tasks, schedule_identity
from comparison_root import common_root

HGM_RULE = 'empirical_mean_then_sample_count_then_candidate_id'
DGM_RULE = 'empirical_mean_then_legacy_archive_order'


def profile(manifest):
    exp = manifest.get('experiment', {})
    algorithm = exp.get('algorithm')
    check((algorithm == 'hgm' and exp.get('hgm_profile') == 'legacy_code') or
          (algorithm == 'dgm' and exp.get('dgm_profile') == 'legacy_code') or
          (algorithm == 'wgm_v2' and exp.get('profile') == 'wgm_exp3_bai_v2'), 'FINAL_UNSUPPORTED_PROFILE')
    return algorithm


def config_from_manifest(manifest, workers=None):
    """Restore recorded execution settings; no mutable default YAML can change a final."""
    from config import HGMConfig, validate_evaluation_workers
    from wgm_v2_config import WGMV2Config
    from dgm_legacy_config import DGMConfig
    cfg = HGMConfig()
    effective = manifest['configuration']['effective']
    for name in ('llm', 'codex', 'execution', 'evaluation', 'paths', 'candidate', 'optimization', 'diagnosis_context'):
        if name in effective:
            if name == 'codex':
                # The recorded provider selects the runner config (Codex when absent).
                from config import codex_config_from_dict
                cfg.codex = codex_config_from_dict(effective['codex'])
            else:
                setattr(cfg, name, type(getattr(cfg, name))(**effective[name]))
    algorithm = profile(manifest)
    cfg.search.algorithm = algorithm
    cfg.search.hgm_profile = cfg.search.dgm_profile = 'legacy_code'
    cfg.search.seed = manifest['experiment'].get('seed') or 42
    if manifest['experiment'].get('incremental'):
        from hgm_incremental_config import HGMIncrementalConfig
        cfg.hgm_incremental = HGMIncrementalConfig()
        cfg.search.seed = manifest['experiment']['seed']
    if algorithm in ('dgm', 'wgm_v2'):
        cls = DGMConfig if algorithm == 'dgm' else WGMV2Config
        allowed = {f.name for f in fields(cls)}
        params = {k:v for k,v in manifest['experiment']['parameters'].items() if k in allowed}
        setattr(cfg, algorithm if algorithm == 'wgm_v2' else 'dgm', cls(**params))
    cfg.execution.evaluation_workers = validate_evaluation_workers(
        workers if workers is not None else cfg.execution.evaluation_workers)
    cfg.execution.policy = 'validated'
    cfg.codex.network_readiness = cfg.network_readiness
    cfg.paths.continue_from = None
    for name in ('dataset_path', 'polyglot_dataset_path'):
        path = Path(getattr(cfg.evaluation, name))
        setattr(cfg.evaluation, name, str(path if path.is_absolute() else ROOT / path))
    cfg.validate_search()
    return cfg


class LegacyCheckpoint:
    def __init__(self, directory, b, frozen=None):
        self.directory = io._checked_path(directory)
        check(type(b) is int and b > 0, 'FINAL_CHECKPOINT_REQUIRED')
        self.b = b
        self.manifest = io._read_json(self.directory / 'run_manifest.json')
        self.algorithm = profile(self.manifest)
        check(self.algorithm in ('hgm', 'dgm'), 'FINAL_LEGACY_REQUIRED')
        self.run = ExecutionRun(self.directory, self.manifest)
        self.output = self.directory / 'derived' / f'checkpoint-{b:06d}'
        path = self.directory / 'checkpoints' / f'{b:06d}.json'
        if self.manifest['experiment'].get('incremental'):
            from hgm_incremental import read_snapshot
            check(path.is_file(), 'FINAL_HGM_SNAPSHOT_REQUIRED')
            read_snapshot(path, self.manifest['reproducibility_fingerprint'])
        self.input_path = path if path.exists() else self.directory / 'search_state.json'
        self.input_row = io._read_json(self.input_path)
        if self.algorithm == 'dgm' and frozen is not None and 'dgm_checkpoint_ref' in frozen:
            # DGM snapshots already exist immutably under checkpoints/. Avoid
            # duplicating their full state twice inside a final manifest.
            check(path.is_file() and frozen == {'dgm_checkpoint_ref': {
                'checkpoint': b, 'sha256': self.input_row['sha256']}}, 'FINAL_DGM_CHECKPOINT_REFERENCE')
            frozen = self.input_row
        self.row = deepcopy(frozen if frozen is not None else self.input_row)
        self.state = self.row['state']
        self.digest = self.row['sha256']
        count = self.state.get('valid_evaluations') if self.algorithm == 'dgm' else self.state.get('n_task_evals')
        check(self.digest == digest(self.state) and count == b and
              self.row['fingerprint'] == self.manifest['reproducibility_fingerprint'], 'FINAL_CHECKPOINT_IDENTITY')
        if frozen is not None and path.exists():
            check(self.row == self.input_row, 'FINAL_CHECKPOINT_CHANGED')
        elif frozen is not None:
            current = self.input_row['state']
            current_count = current.get('valid_evaluations', current.get('n_task_evals'))
            check(current_count > b or self.row == self.input_row, 'FINAL_CHECKPOINT_CHANGED')
        if self.algorithm == 'hgm':
            check(self.state['experiment'] == self.manifest['experiment'], 'FINAL_CHECKPOINT_PROFILE')
        else:
            self.dgm_candidates = self._dgm_candidates()
        from task_schedule import manifest_polyglot
        self.final_tasks = canonical_tasks(ROOT, 42, polyglot=manifest_polyglot(self.manifest))
        self.execution = self.state if self.algorithm == 'dgm' else self.state['execution']
        consumed = self.execution['consumed_bindings']
        check(len(consumed) == b, 'FINAL_SEARCH_BUDGET_BINDINGS')
        if self.algorithm == 'hgm':
            check(set(self.execution['observations']) == set(consumed), 'FINAL_DIRECT_OBSERVATIONS_CHANGED')
        plans = {p.identity.evaluation_id:p for p in self.run.plans.read()}
        self.direct = {}
        for eid, binding in consumed.items():
            check(eid in plans, 'FINAL_SEARCH_PLAN_MISSING')
            p = plans[eid]
            r = read_binding(self.run.ledger, p)
            check(r.binding_digest == binding and r.outcome == 'success' and p.identity.stage == 'search',
                  'FINAL_SEARCH_BINDING_CHANGED')
            rows = self.direct.setdefault(p.identity.candidate_id, {})
            check(p.identity.task_id not in rows, 'FINAL_DUPLICATE_DIRECT_TASK')
            rows[p.identity.task_id] = int(r.resolved)
            if self.algorithm == 'hgm':
                check(self.execution['observations'][eid] == int(r.resolved), 'FINAL_DIRECT_OBSERVATIONS_CHANGED')
        # Validated non-import HGM root evidence is also direct and outside search budget.
        roots = [p for p in plans.values() if p.identity.stage == 'root']
        for p in roots:
            r = read_binding(self.run.ledger, p)
            check(p.identity.candidate_id == 'initial' and r.outcome == 'success', 'FINAL_ROOT_BINDING')
            self.direct.setdefault('initial', {})[p.identity.task_id] = int(r.resolved)

    def unchanged(self):
        check(io._read_json(self.input_path) == self.input_row, 'FINAL_CANONICAL_CHANGED')

    def _dgm_candidates(self):
        """Validate frozen generation identities without admitting/selecting any child."""
        from search_operation_journal import SearchOperationPlan, validate_rng
        state, manifest = self.state, self.manifest
        check(type(state.get('dgm_schema')) is int and state['dgm_schema'] == 1
              and manifest['run_type'] == 'dgm' and state['summary']['algorithm'] == 'dgm'
              and manifest['configuration']['effective']['search'] == manifest['experiment']
              and manifest['configuration']['sha256'] == digest(manifest['configuration']['effective']),
              'FINAL_DGM_SCHEMA')
        validate_rng(state['rng'])
        archive, known, committed = ['initial'], {'initial'}, 0
        for index, generation in enumerate(state['generations']):
            check(generation['index'] == index and generation['archive_before'] == archive
                  and type(generation['committed']) is bool
                  and (generation['committed'] or index == len(state['generations'])-1),
                  'FINAL_DGM_ARCHIVE_CHANGED')
            children = {}
            for child in generation['children']:
                cid = child['id']
                plan = SearchOperationPlan.from_dict(child['operation_plan'])
                identity = plan.identity
                check(cid not in known and identity['candidate_id'] == cid
                      and identity['algorithm'] == 'dgm' and identity['profile'] == 'legacy_code'
                      and identity['run_id'] == manifest['run_id']
                      and identity['run_fingerprint'] == manifest['reproducibility_fingerprint']
                      and identity['parent']['candidate_id'] == child['parent'] in archive
                      and type(child['admitted']) is bool, 'FINAL_DGM_CANDIDATE_IDENTITY')
                known.add(cid)
                children[cid] = child
                if child['admitted']:
                    check(generation['committed'] and child['status'] == 'evaluated'
                          and child['operation_terminal']['outcome'] == 'candidate'
                          and 'small' in child['stages'] and all(
                              stage['cursor'] == stage['scheduled'] == len(stage['plans'])
                              for stage in child['stages'].values()), 'FINAL_DGM_ADMISSION_CHANGED')
            order = generation['completion_order']
            check(len(order) == len(set(order)) and set(order) <= set(children), 'FINAL_DGM_ARCHIVE_CHANGED')
            admitted = [cid for cid in order if children[cid]['admitted']]
            check(generation['admitted'] == admitted, 'FINAL_DGM_ADMISSION_CHANGED')
            if generation['committed']:
                check(set(order) == set(children) and generation['rejected'] == [
                    cid for cid in order if not children[cid]['admitted']], 'FINAL_DGM_ARCHIVE_CHANGED')
                archive.extend(admitted)
                committed += 1
            else:
                check(not generation['admitted'] and not generation['rejected'], 'FINAL_DGM_ADMISSION_CHANGED')
        check(state['archive'] == archive and len(archive) == len(set(archive))
              and state['node_ids'] == {cid:i for i,cid in enumerate(archive)}
              and all(type(v) is int for v in state['node_ids'].values())
              and set(archive) <= set(state['metadata']) <= known
              and state['generation'] == committed and state['summary']['archive_size'] == len(archive),
              'FINAL_DGM_ARCHIVE_CHANGED')
        return known

    def sources_checkpoint(self):
        if self.algorithm == 'hgm':
            return self
        return SimpleNamespace(directory=self.directory, manifest=self.manifest, state={
            'nodes': [{'commit_id': c} for c in self.state['archive']],
            'execution': {'consumed_operations': self.state['consumed_operations']}})

    def observations(self, cached):
        rows = deepcopy(self.direct)
        imported = (self.algorithm == 'dgm' or self.manifest.get('root_initialization', {}).get('comparison_import'))
        if imported:
            check('initial' not in rows, 'FINAL_ROOT_EVIDENCE_CONFLICT')
            rows['initial'] = {r['task_id']: int(r['result'] == 'resolved') for r in cached['per_task_results']}
        if self.algorithm == 'hgm':
            ids = [n['commit_id'] for n in self.state['nodes']]
            expected = {c: rows.get(c, {}) for c in ids}
            check({c:r for c,r in self.state['results'].items() if r} ==
                  {c:r for c,r in expected.items() if r}, 'FINAL_DIRECT_OBSERVATIONS_CHANGED')
            check(len(ids) == len(set(ids)) and set(rows) <= (set(ids) | set(self.state.get('node_ids', {}))),
                  'FINAL_CHECKPOINT_NODES')
        else:
            ids = self.state['archive']
            # Bound observations may belong to a child of an uncommitted
            # generation. node_ids contains only admitted archive members.
            check(set(rows) <= self.dgm_candidates, 'FINAL_CHECKPOINT_NODES')
            for c in ids:
                perf = self.state['metadata'][c]['overall_performance']
                own = rows.get(c, {})
                submitted = perf['total_submitted_ids']
                resolved = perf['total_resolved_ids']
                check(bool(own) and len(submitted) == len(set(submitted)) == len(own)
                      and set(submitted) == set(own) and len(resolved) == len(set(resolved))
                      and set(resolved) == {t for t,v in own.items() if v}
                      and perf['total_submitted_instances'] == len(own)
                      and perf['total_resolved_instances'] == sum(own.values())
                      and perf['accuracy_score'] == sum(own.values()) / len(own), 'FINAL_DGM_PERFORMANCE_CHANGED')
            rows = {c: rows[c] for c in ids}
            selection_path = self.directory / 'checkpoints' / f'{self.b:06d}.selection.json'
            if selection_path.exists():
                saved = io._read_json(selection_path)
                selected = saved['candidate_id']
                check(selected in ids, 'FINAL_DGM_SELECTION_CHANGED')
                wins, count = sum(rows[selected].values()), len(rows[selected])
                # Validate the frozen winner, without invoking a selector on resume.
                check(all(sum(rows[c].values()) * count <= wins * len(rows[c])
                          and (i >= ids.index(selected) or sum(rows[c].values()) * count < wins * len(rows[c]))
                          for i,c in enumerate(ids)), 'FINAL_DGM_SELECTION_CHANGED')
                check(saved == {'checkpoint_digest': self.digest,
                    'final_selection_rule': DGM_RULE, 'candidate_id': selected, 'archive': ids,
                    'performance': self.state['metadata'][selected]['overall_performance']},
                    'FINAL_DGM_SELECTION_CHANGED')
        return ids, rows


def choose_legacy(algorithm, ids, direct):
    eligible = [c for c in ids if direct.get(c)]
    check(bool(eligible), 'FINAL_NO_DIRECT_OBSERVATIONS')
    from fractions import Fraction
    def mean(c):
        return Fraction(sum(direct[c].values()), len(direct[c]))
    chosen = (min(eligible, key=lambda c: (-mean(c), -len(direct[c]), c.encode('utf-8')))
              if algorithm == 'hgm' else max(eligible, key=mean))
    return {'candidate_id': chosen, 'direct_observations': len(direct[chosen]),
            'direct_successes': sum(direct[chosen].values()), 'empirical_mean': float(mean(chosen)),
            'final_selection_rule': HGM_RULE if algorithm == 'hgm' else DGM_RULE,
            'final_selection_source': 'external_comparison_adapter' if algorithm == 'hgm' else 'checkpoint_archive'}


def _source_identity(source):
    return {k:source[k] for k in ('source_fingerprint', 'policy_fingerprint', 'chain_digest')}


def selection_evidence(cp, selected, source):
    selection = {**{k:v for k,v in selected.items() if k != 'selection_digest'},
                 **_source_identity(source), 'checkpoint_digest': cp.digest,
                 'checkpoint': cp.b, 'valid_search_budget': cp.b}
    selection['selection_digest'] = digest(selection)
    return selection


def _legacy_manifest(cp, config, entries, selected, source, certificate, root_path):
    row = deepcopy(cp.manifest)
    selection = selection_evidence(cp, selected, source)
    if 'final_selection' in cp.row:
        check(selection == cp.row['final_selection'], 'FINAL_FROZEN_SELECTION_CHANGED')
    row['comparison_final'] = {'schema_version': 1, 'algorithm': cp.algorithm, 'profile': 'legacy_code',
        'canonical_run_id': cp.manifest['run_id'], 'canonical_fingerprint': cp.manifest['reproducibility_fingerprint'],
        'checkpoint_snapshot': ({'dgm_checkpoint_ref': {'checkpoint': cp.b, 'sha256': cp.digest}}
            if cp.algorithm == 'dgm' and cp.input_path == cp.directory / 'checkpoints' / f'{cp.b:06d}.json'
            else cp.row), 'selection': selection, 'common_root': certificate,
        'root_cache_path': str(root_path), 'entry_hashes': {t:digest(entries[t]) for t in cp.final_tasks},
        'search_accounting': search_accounting(cp)}
    row['run_id'] = 'final-' + digest(row['comparison_final'])
    row['tasks'].update(ordered_ids=cp.final_tasks, count=len(cp.final_tasks), sha256=digest(cp.final_tasks),
                        **schedule_identity(cp.final_tasks, 42))
    from evaluation_batch import runtime_workers
    row['evaluation_parallelism'] = runtime_workers(config, cp.algorithm, len(cp.final_tasks), domain='primary')
    row['evaluation_execution'] = execution_identity(config, row)
    row['fingerprint_inputs'] = fingerprint_inputs(row)
    row['reproducibility_fingerprint'] = digest(row['fingerprint_inputs'])
    return row


def search_accounting(cp):
    from wgm_v2_derived import accounting
    state = cp.state
    context = state if cp.manifest['experiment']['algorithm'] == 'dgm' else state['execution']
    # Include pending attempts frozen at this checkpoint, but never later search work.
    plans = cp.run.plans.read() if hasattr(cp, 'run') else ExecutionRun(cp.directory, cp.manifest).plans.read()
    eval_ids = set(context['consumed_bindings'])
    op_ids = set(context.get('consumed_operations', {}))
    if cp.manifest['experiment']['algorithm'] == 'dgm':
        from evaluation_plan import EvaluationPlan
        for gen in state['generations']:
            for child in gen['children']:
                op_ids.add(child['operation_plan']['operation_id'])
                for stage in child['stages'].values():
                    eval_ids.update(EvaluationPlan.from_dict(p).identity.evaluation_id
                                    for p in stage['plans'][:stage['scheduled']])
    else:
        inflight = state.get('inflight') or {}
        if 'evaluation_plan' in inflight:
            eval_ids.add(inflight['evaluation_plan']['identity']['evaluation_id'])
        # Operation plans carry immutable expansion ordinals.
        from search_operation_journal import SearchOperationJournal
        if (cp.directory / 'search_operations/head.json').exists():
            journal = SearchOperationJournal(cp.directory, cp.manifest['run_id'], cp.manifest['reproducibility_fingerprint'])
            op_ids.update(p.operation_id for p in journal.plans() if p.identity['ordinal'] < len(state['history']))
    stats = accounting(cp.directory, evaluation_ids=eval_ids, operation_ids=op_ids)
    stats['downstream_model_calls'] = stats['physical_model_calls'] - stats['diagnosis_calls'] - stats['generation_calls']
    stats['evaluator_calls'] = stats['physical_evaluator_calls']
    # WGM's search stage ends below b (B_fin is reserved inside b); others spend exactly b.
    valid = state['n_task_evals'] if cp.manifest['experiment']['algorithm'] == 'wgm_v2' else cp.b
    wall = cp.row.get('search_wall_time_seconds')
    latest_path = cp.directory / 'search_state.json'
    if latest_path.exists() and wall is None:
        latest = io._read_json(latest_path)['state']
        count = latest.get('valid_evaluations', latest.get('n_task_evals'))
        summary = cp.directory / 'run_summary.json'
        if count == valid and summary.exists():
            wall = io._read_json(summary).get('timing', {}).get('whole_run_total', wall)
    return {**stats, 'search_wall_seconds': wall, 'valid_search_evaluations': valid,
            'search_schedule_hash': cp.manifest['tasks']['task_order_hash'],
            'workers': cp.manifest.get('evaluation_parallelism')}


def _binding_report(directory, tasks, candidate, selection_digest, source):
    """Reconstruct results from authoritative plans and bindings, never trust summary counts."""
    from evaluation_batch import batch_reports, _ledger_counts
    from wgm_v2_derived import accounting
    manifest = io._read_json(directory / 'run_manifest.json')
    run = ExecutionRun(directory, manifest)
    plans = run.plans.read()
    check(len(plans) <= len(tasks), 'FINAL_PLAN_COUNT')
    check(manifest['tasks']['ordered_ids'] == tasks and all(
        s.identity.evaluation_id in {p.identity.evaluation_id for p in plans} for s in run.ledger.evaluations()),
        'FINAL_UNCLAIMED_EVALUATION')
    for index, p in enumerate(plans):
        check(p.identity.stage == 'final_benchmark' and p.identity.ordinal == index
              and p.identity.task_id == tasks[index] and p.identity.candidate_id == candidate
              and p.selection_digest == selection_digest and
              all(getattr(p.identity, k) == source[k] for k in ('source_fingerprint', 'policy_fingerprint')),
              'FINAL_PLAN_CHANGED')
        inputs = manifest.get('comparison_final', manifest.get('v2_derived', {}))['entry_hashes']
        check(p.identity.task_input_fingerprint == inputs[p.identity.task_id], 'FINAL_TASK_INPUT_CHANGED')
    counts = _ledger_counts(run, plans) if plans else dict(completed=0, failed=0, ambiguous=0, results=[])
    outcomes, evidence, errors = {}, [], []
    for p in plans:
        inspection = run.ledger.inspect(p.identity.evaluation_id, p.identity)
        check(not inspection.error_code, 'FINAL_LEDGER_CHANGED')
        if inspection.snapshot and inspection.snapshot.state == 'bound':
            r = read_binding(run.ledger, p)
            evidence.append({'evaluation_id': p.identity.evaluation_id, 'binding_digest': r.binding_digest})
            if r.outcome == 'success':
                outcomes[p.identity.task_id] = r.resolved
            else:
                errors.append(p.identity.task_id)
        elif ('BLOCK_' in io.classify(inspection).value or 'FAILED' in io.classify(inspection).value
              or io.classify(inspection).value == 'RETRY_PREPARATION'):
            errors.append(p.identity.task_id)
    batches = batch_reports(directory)
    timing = accounting(directory)
    parallel = manifest.get('evaluation_parallelism') or {}
    return {**_outcomes(tasks, outcomes), 'submitted': timing['evaluation_model_markers'], 'error_ids': errors,
        'ambiguous': counts['ambiguous'], 'complete': len(outcomes) == len(tasks),
        'status': 'completed' if len(outcomes) == len(tasks) else 'ambiguous' if counts['ambiguous'] else 'incomplete',
        'configured_workers': parallel.get('configured_workers'), 'effective_workers': parallel.get('effective_workers'),
        'observed_workers': max((x.get('peak_concurrency', 0) for x in batches), default=0),
        'worker_history': [s for x in batches for s in x['sessions']],
        'final_physical_model_calls': timing['physical_model_calls'],
        'final_physical_evaluator_calls': timing['physical_evaluator_calls'],
        'task_duration_sum_seconds': sum(x.get('cumulative_task_duration_sum_seconds', 0) for x in batches),
        'batch_wall_seconds': sum(x.get('cumulative_batch_wall_seconds', 0) for x in batches),
        'final_wall_seconds': sum(x.get('cumulative_batch_wall_seconds', 0) for x in batches),
        'reused_verified_root': False, 'evidence': evidence}


def _outcomes(tasks, outcomes):
    return {'requested': len(tasks), 'completed': len(outcomes), 'resolved': sum(outcomes.values()),
        'accuracy': sum(outcomes.values()) / len(tasks) if len(outcomes) == len(tasks) else None,
        'resolved_ids': [t for t in tasks if outcomes.get(t) is True],
        'unresolved_ids': [t for t in tasks if outcomes.get(t) is False]}


def _seal_report(row):
    row['final_artifacts_digest'] = digest(row)
    return row


def legacy_report(directory, b):
    folder = Path(directory) / 'derived' / f'checkpoint-{b:06d}' / 'final'
    manifest = io._read_json(folder / 'run_manifest.json')
    run = ExecutionRun(folder, manifest)
    adapter = manifest['comparison_final']
    cp = LegacyCheckpoint(directory, b, adapter['checkpoint_snapshot'])
    selected = adapter['selection']
    if 'final_selection' in cp.row:
        check(selected == cp.row['final_selection'], 'FINAL_FROZEN_SELECTION_CHANGED')
    check(selected['selection_digest'] == digest({k:v for k,v in selected.items() if k != 'selection_digest'})
          and selected['checkpoint_digest'] == cp.digest and selected['checkpoint'] == b
          and adapter['canonical_fingerprint'] == cp.manifest['reproducibility_fingerprint'], 'FINAL_SELECTION_CHANGED')
    certificate, cached, _ = common_root(cp.manifest, adapter['root_cache_path'])
    check(certificate == adapter['common_root'], 'FINAL_ROOT_CHANGED')
    ids, _ = cp.observations(cached)
    check(selected['candidate_id'] in ids, 'FINAL_CANDIDATE_NOT_IN_CHECKPOINT')
    from wgm_v2_derived import CheckpointSources
    source = CheckpointSources(cp.sources_checkpoint(), folder).get(selected['candidate_id'], read_only=True)
    check(_source_identity(source) == _source_identity(selected), 'FINAL_SOURCE_CHANGED')
    reuse_path = folder / 'root_reuse.json'
    if reuse_path.exists():
        reuse = io._read_json(reuse_path)
        check(selected['candidate_id'] == 'initial' and not run.plans.read()
              and not run.ledger.evaluations() and reuse == {'selection_digest': selected['selection_digest'],
                'root_cache_digest': certificate['root_cache_digest'], 'evidence_digest': digest(cached['per_task_results'])},
              'FINAL_ROOT_REUSE_CHANGED')
        outcomes = {r['task_id']: r['result'] == 'resolved' for r in cached['per_task_results']}
        result = {**_outcomes(cp.final_tasks, outcomes), 'submitted': 0, 'error_ids': [], 'ambiguous': 0,
            'complete': True, 'status': 'completed', 'reused_verified_root': True,
            'configured_workers': manifest['evaluation_parallelism']['configured_workers'],
            'effective_workers': 0, 'observed_workers': 0, 'worker_history': [],
            'final_physical_model_calls': 0, 'final_physical_evaluator_calls': 0,
            'task_duration_sum_seconds': 0, 'batch_wall_seconds': 0, 'final_wall_seconds': 0, 'evidence': reuse}
    else:
        result = _binding_report(folder, cp.final_tasks, selected['candidate_id'], selected['selection_digest'], selected)
    cp.unchanged()
    return _seal_report({**result, **selected, 'algorithm': cp.algorithm, 'profile': 'legacy_code',
        'role': 'primary', 'search_profile': 'legacy_code', 'common_root': certificate,
        'task_membership_hash': certificate['task_membership_hash'], 'task_order_hash': certificate['task_order_hash'],
        'execution_policy': cp.manifest['execution_policy']['effective_policy'],
        'search': adapter['search_accounting'], 'formal_comparison_eligible': _formal(cp),
        'derived_fingerprint': manifest['reproducibility_fingerprint'], 'secondary': None})


def _formal(cp):
    if cp.manifest['experiment']['algorithm'] == 'dgm':
        return cp.manifest['dgm_runtime']['formal_comparison_eligible'] is True
    if cp.manifest['experiment']['algorithm'] == 'wgm_v2':
        return cp.row['runtime_control']['protocol'] == 'formal'
    root = cp.manifest.get('root_initialization', {})
    return root.get('comparison_import') is True and root.get('comparison_protocol') == 'formal'


def run_legacy(config, directory, b, *, root_cache=None, preflight, execute_factory=None, sources_factory=None,
               retry_preparation=None):
    from validated_runs import _entry_map, executor, fixed_tasks
    from wgm_v2_derived import CheckpointSources
    from run_metadata import RunRecorder, repository_snapshot, check_repository
    folder = Path(directory) / 'derived' / f'checkpoint-{b:06d}' / 'final'
    saved = io._read_json(folder / 'run_manifest.json') if folder.exists() else None
    cp = LegacyCheckpoint(directory, b, saved['comparison_final']['checkpoint_snapshot'] if saved else None)
    from execution_policy import require_supported
    from evaluation_plan import model_fingerprint
    require_supported(config)
    check(model_fingerprint(config) == cp.manifest['evaluation_execution']['model_fingerprint'], 'FINAL_MODEL_CHANGED')
    certificate, cached, root_path = common_root(cp.manifest, root_cache or (
        saved['comparison_final']['root_cache_path'] if saved else None))
    if cp.manifest.get('hgm_incremental_inputs'):
        check(certificate == cp.manifest['hgm_incremental_inputs']['common_root'], 'FINAL_ROOT_CHANGED')
    ids, direct = cp.observations(cached)
    if saved:
        run = ExecutionRun(folder, saved)
        selected = saved['comparison_final']['selection']
        check(saved['comparison_final']['common_root'] == certificate and selected['candidate_id'] in ids,
              'FINAL_FROZEN_SELECTION_CHANGED')
    else:
        selected = cp.row['final_selection'] if 'final_selection' in cp.row else choose_legacy(cp.algorithm, ids, direct)
    # Sources only materialize under derived. Initial reconstruction must not pre-create the final ledger directory.
    sources = (sources_factory or CheckpointSources)(cp.sources_checkpoint(), folder)
    if saved and (folder / 'summary.json').exists():
        result = legacy_report(directory, b)
        if result['complete']:
            return result
    entries = _entry_map(config, cp.final_tasks)
    expected_hashes = cp.manifest.get('dgm_inputs', {}).get('entry_hashes')
    if expected_hashes:
        check(all(digest(entries[t]) == expected_hashes[t] for t in cp.final_tasks), 'FINAL_TASK_INPUT_CHANGED')
    # A temporary derived cache allows the ledger to own the fresh final directory.
    staging_sources = (sources_factory or CheckpointSources)(cp.sources_checkpoint(), cp.output / 'source-cache')
    source = staging_sources.get(selected['candidate_id'])
    if not saved:
        manifest = _legacy_manifest(cp, config, entries, selected, source, certificate, root_path)
        folder.parent.mkdir(parents=True, exist_ok=True)
        run = ExecutionRun.create(folder, manifest)
    selected = run.manifest['comparison_final']['selection']
    with run.lock():
        source = sources.get(selected['candidate_id'])
        check(_source_identity(source) == _source_identity(selected), 'FINAL_SOURCE_CHANGED')
        check({t:digest(entries[t]) for t in cp.final_tasks} == run.manifest['comparison_final']['entry_hashes'],
              'FINAL_TASK_INPUT_CHANGED')
        if selected['candidate_id'] == 'initial':
            check(source['source_fingerprint'] == certificate['source_digest'] and
                  source['policy_fingerprint'] == certificate['policy_digest'] and not run.plans.read(), 'FINAL_ROOT_REUSE_IDENTITY')
            reuse = {'selection_digest': selected['selection_digest'],
                'root_cache_digest': certificate['root_cache_digest'], 'evidence_digest': digest(cached['per_task_results'])}
            reuse_path = folder / 'root_reuse.json'
            if reuse_path.exists():
                check(io._read_json(reuse_path) == reuse, 'FINAL_ROOT_REUSE_CHANGED')
            else:
                durable_json(reuse_path, reuse)
        else:
            if execute_factory is None:
                repository = repository_snapshot(ROOT)
                check_repository(repository)
                from preparation_recovery import check_implementation
                check_implementation(run, ROOT, cp.manifest['repository']['commit'], repository, retry_preparation)
            elif retry_preparation:
                from preparation_recovery import authorize
                authorize(run, retry_preparation, {'old_commit': cp.manifest['repository']['commit'],
                    'new_commit': cp.manifest['repository']['commit'], 'reason': 'injected_test_executor'})
            run.precheck()
            preflight(config, output_root=folder)
            recorder = RunRecorder(folder, run.manifest, network_readiness=config.network_readiness)
            cancel = threading.Event()
            execute = (execute_factory or executor)(config, run, entries, sources, recorder, cancel_event=cancel)
            try:
                fixed_tasks(run, entries, selected['candidate_id'], source, execute, stage='final_benchmark',
                            tasks=cp.final_tasks, selection_digest=selected['selection_digest'],
                            workers=config.execution.evaluation_workers, cancel_event=cancel)
            finally:
                cp.unchanged()
                durable_json(folder / 'summary.json', legacy_report(directory, b), replace_existing=True)
        cp.unchanged()
        result = legacy_report(directory, b)
        durable_json(folder / 'summary.json', result, replace_existing=True)
        return result


def wgm_report(directory, b, role='primary'):
    from wgm_v2_derived import Checkpoint, CheckpointSources
    cp = Checkpoint(directory, b)
    context = io._read_json(cp.output / 'comparison_context.json')
    check(context['sha256'] == digest(context['identity']) and
          context['identity']['checkpoint_digest'] == cp.digest, 'FINAL_WGM_CONTEXT_CHANGED')
    context = context['identity']
    certificate, _, _ = common_root(cp.manifest, context['root_cache_path'])
    check(certificate == context['common_root'], 'FINAL_ROOT_CHANGED')
    selected = cp.selection[role]
    check(selected is not None, 'DERIVED_SECONDARY_UNAVAILABLE')
    folder = cp.output / role
    manifest = io._read_json(folder / 'run_manifest.json')
    derived = manifest['v2_derived']
    check(derived['canonical_checkpoint_digest'] == cp.digest and derived['domain'] == role
          and derived['selection'] == selected and manifest['tasks']['ordered_ids'] == cp.final_tasks,
          'FINAL_WGM_SELECTION_CHANGED')
    source = _source_identity(CheckpointSources(cp, folder).get(selected['candidate_id'], read_only=True))
    result = _binding_report(folder, cp.final_tasks, selected['candidate_id'], digest(derived), source)
    state = io._read_json(folder / 'derived_state.json')
    check(state['sha256'] == digest(state['state']) and state['state']['identity'] == digest(derived),
          'FINAL_WGM_STATE_CHANGED')
    # Existing derived consumption must agree with the same authoritative binding prefix.
    records = state['state']['records']
    evidence = result['evidence']
    check(len(records) <= len(evidence) and all(
        r['plan']['identity']['evaluation_id'] == e['evaluation_id'] and r['binding_digest'] == e['binding_digest']
        for r,e in zip(records, evidence)), 'FINAL_WGM_BINDING_CHANGED')
    cp.unchanged()
    return _seal_report({**result, **source, 'algorithm': 'wgm_v2', 'profile': 'wgm_v2', 'role': role,
        'checkpoint': b, 'valid_search_budget': b, 'checkpoint_digest': cp.digest,
        'candidate_id': selected['candidate_id'], 'selection_digest': digest(derived),
        'final_selection_rule': 'existing_checkpoint_primary' if role == 'primary' else 'existing_checkpoint_secondary',
        'final_selection_source': 'wgm_v2_checkpoint', 'selection': selected,
        'common_root': certificate, 'task_membership_hash': certificate['task_membership_hash'],
        'task_order_hash': certificate['task_order_hash'],
        'execution_policy': cp.manifest['execution_policy']['effective_policy'],
        'search': context['search_accounting'], 'formal_comparison_eligible': _formal(cp),
        'derived_fingerprint': manifest['reproducibility_fingerprint']})


def final_report(directory, b):
    manifest = io._read_json(Path(directory) / 'run_manifest.json')
    if profile(manifest) != 'wgm_v2':
        return legacy_report(directory, b)
    base = Path(directory) / 'derived' / f'checkpoint-{b:06d}'
    primary = wgm_report(directory, b) if (base / 'primary' / 'run_manifest.json').exists() else None
    secondary = (wgm_report(directory, b, 'secondary') if (base / 'secondary' / 'run_manifest.json').exists()
                 else None)
    return {'algorithm': 'wgm_v2', 'checkpoint': b, 'primary': primary, 'secondary': secondary}


def run_final(directory, b, *, workers=None, roles='primary-only', root_cache=None, readiness=None, preflight,
              retry_preparation=None):
    manifest = io._read_json(Path(directory) / 'run_manifest.json')
    config = config_from_manifest(manifest, workers)
    if readiness is not None:
        config.network_readiness = readiness
        config.codex.network_readiness = readiness
    if profile(manifest) != 'wgm_v2':
        check(roles == 'primary-only', 'LEGACY_FINAL_SINGLE_AGENT_ONLY')
        return run_legacy(config, directory, b, root_cache=root_cache, preflight=preflight,
                          retry_preparation=retry_preparation)
    check(retry_preparation is None or roles in ('primary-only', 'secondary-only'),
          'PREPARATION_RETRY_SINGLE_ROLE_REQUIRED')
    from wgm_v2_derived import Checkpoint, run_derived
    cp = Checkpoint(directory, b)
    certificate, _, cache_path = common_root(manifest, root_cache)
    context_path = cp.output / 'comparison_context.json'
    if context_path.exists():
        context = io._read_json(context_path)
        check(context['sha256'] == digest(context['identity']) and
              context['identity']['checkpoint_digest'] == cp.digest and
              context['identity']['common_root'] == certificate, 'FINAL_WGM_CONTEXT_CHANGED')
    else:
        body = {'checkpoint_digest': cp.digest, 'common_root': certificate, 'root_cache_path': str(cache_path),
                'search_accounting': search_accounting(cp)}
        cp.output.mkdir(parents=True, exist_ok=True)
        durable_json(context_path, {'identity': body, 'sha256': digest(body)})
    # Existing WGM execution, selection identity, schemas and optional Secondary are unchanged.
    run_derived(config, directory, b, 'final', roles=roles, preflight=preflight,
                **({'retry_preparation': retry_preparation} if retry_preparation else {}))
    result = final_report(directory, b)
    durable_json(cp.output / 'comparison_final.json', result, replace_existing=True)
    return result


def final_text(row):
    if row.get('algorithm') == 'wgm_v2' and 'primary' in row:
        return '\n\n'.join(role + ': ' + (final_text(row[role]) if row[role] else 'not executed')
                           for role in ('primary', 'secondary'))
    return '\n'.join((f"{row['algorithm']}/{row['profile']} checkpoint={row['checkpoint']} status={row['status']}",
        f"Selected: {row['candidate_id']} ({row['final_selection_rule']})",
        f"Final60: {row['resolved']}/{row['completed']} completed; accuracy={row['accuracy']}; reused root={row['reused_verified_root']}",
        f"Calls: model={row['final_physical_model_calls']}, evaluator={row['final_physical_evaluator_calls']}",
        f"Workers: configured={row['configured_workers']}, effective={row['effective_workers']}, observed={row['observed_workers']}",
        f"Final wall seconds: {row['final_wall_seconds']}; artifacts={row['final_artifacts_digest']}"))
