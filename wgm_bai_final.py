"""Explicit checkpoint BAI -> final60, separate from canonical Primary ablations."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import threading

import evaluation_ledger as io
from evaluation_plan import check, durable_json, execution_identity
from evaluation_scheduler import ExecutionRun
from run_metadata import digest, fingerprint_inputs
from wgm_v2_derived import (Checkpoint, CheckpointSources, DerivedEvaluation, make_manifest, accounting)
from final_adapter import (config_from_manifest, _binding_report, _source_identity, _seal_report,
                           _formal, search_accounting, wgm_report)
from comparison_root import common_root

FINAL = 'bai-final'
RULE = 'maximum_lower_bound_then_candidate_utf8_bytes'


class ReadOnlySources:
    def __init__(self, cp, folder):
        self.sources, self.cache = CheckpointSources(cp, folder), {}

    def get(self, candidate):
        if candidate not in self.cache:
            self.cache[candidate] = self.sources.get(candidate, read_only=True)
        return self.cache[candidate]


def read_bai(cp, *, entries=None, require_complete=False):
    """Replay deterministic decisions against existing bindings, without publication."""
    from validated_runs import _entry_map
    folder = cp.output / 'bai'
    budget = int(cp.state['experiment']['parameters']['checkpoint_bai_fraction'] * cp.b)
    result = {'status': 'not_created', 'budget': budget, 'completed': 0, 'planned': 0,
              'failed': 0, 'ambiguous': 0, 'selection': None, 'selection_digest': None,
              'evidence_digest': None, 'physical_model_calls': 0, 'physical_evaluator_calls': 0,
              'wall_seconds': 0.}
    if not folder.exists():
        check(not require_complete, 'BAI_SELECTION_NOT_COMPLETE')
        return result
    config = config_from_manifest(cp.manifest)
    entries = entries if entries is not None else _entry_map(config, cp.pool)
    manifest = io._read_json(folder / 'run_manifest.json')
    expected = make_manifest(cp, config, 'bai', entries)
    check(manifest['fingerprint_inputs'] == fingerprint_inputs(expected), 'BAI_MANIFEST_CHANGED')
    run = ExecutionRun(folder, manifest)
    plans = run.plans.read()
    engine = DerivedEvaluation(cp, run, entries, ReadOnlySources(cp, folder), None)
    if engine.path.exists():
        state = engine.load()
    else:
        check(not plans and not (folder / 'selection.json').exists(), 'BAI_STATE_MISSING')
        state = engine.initial()
    check(len(state['records']) <= len(plans) <= len(state['records']) + int(state['pending'] is not None), 'BAI_PLAN_COUNT')
    snapshots = run.ledger.evaluations()
    check(all(s.identity.evaluation_id in {p.identity.evaluation_id for p in plans} for s in snapshots),
          'BAI_UNCLAIMED_EVALUATION')
    for p in plans:
        inspection = run.ledger.inspect(p.identity.evaluation_id, p.identity)
        check(not inspection.error_code and not inspection.legacy and not inspection.read_only, 'BAI_LEDGER_IDENTITY')
    choice, reason = engine.decision(state)
    done = reason is not None and state['pending'] is None
    selection = None
    if (folder / 'selection.json').exists():
        selection = io._read_json(folder / 'selection.json')
        expected_selection = {'schema_version': 1, 'domain': 'bai', 'checkpoint_digest': cp.digest,
            'completed': len(state['records']), 'requested': budget, 'pending': 0, 'status': 'completed',
            'stop_reason': reason, 'canonical_feedback': False, 'task_level_held_out': False, 'D_design': None,
            'selection': choice['leader'], 'challenger': choice['challenger'],
            'certificate': 'conditional_epsilon' if reason == 'epsilon_satisfied' else None,
            'confidence_scope': 'one_IID_stationary_trajectory_not_joint_checkpoints',
            'draws': [{k: r['plan']['identity'][k] for k in ('task_id', 'evaluation_id', 'candidate_id')}
                      for r in state['records']], 'canonical_selection_overwritten': False}
        check(done and selection == expected_selection, 'BAI_SELECTION_CHANGED')
    stats = accounting(folder)
    result.update(status='completed' if selection else 'incomplete' if plans else 'planned',
        completed=len(state['records']), planned=len(plans),
        failed=sum(s.state == 'bound' and next(io._decode(e.data)['outcome'] for e in reversed(s.events)
                   if e.kind == 'bound') != 'success' for s in snapshots),
        ambiguous=stats['ambiguous_evaluations'],
        selection=selection['selection'] if selection else None,
        selection_digest=digest(selection) if selection else None,
        evidence_digest=digest({'manifest': manifest['reproducibility_fingerprint'], 'state': state,
            'plans': [asdict(p) for p in plans], 'heads': [[s.identity.evaluation_id, s.binding_id,
                s.events[-1].sha256] for s in snapshots], 'selection': selection}),
        physical_model_calls=stats['physical_model_calls'], physical_evaluator_calls=stats['physical_evaluator_calls'])
    summary = folder / 'run_summary.json'
    result['wall_seconds'] = (io._read_json(summary).get('timing', {}).get('whole_run_total')
                              if summary.exists() else None)
    check(not require_complete or result['status'] == 'completed' and not result['failed'] and not result['ambiguous'],
          'BAI_SELECTION_NOT_COMPLETE')
    cp.unchanged()
    return result


def final_manifest(cp, config, entries, bai):
    check(bai['status'] == 'completed' and bai['selection_digest'] and bai['evidence_digest'], 'BAI_SELECTION_NOT_COMPLETE')
    row = make_manifest(cp, config, 'primary', entries)
    row['v2_derived'].update(domain='bai_final', selection=deepcopy(bai['selection']),
        selection_source='checkpoint_bai', selection_rule=RULE,
        bai_selection_digest=bai['selection_digest'], bai_evidence_digest=bai['evidence_digest'])
    row['v2_derived']['verified_root'] = common_root(cp.manifest)[0]
    row['v2_derived']['selected_source'] = _source_identity(CheckpointSources(cp, cp.directory).get(
        bai['selection']['candidate_id'], read_only=True))
    row['run_id'] = 'derived-' + digest(row['v2_derived'])
    row['evaluation_execution'] = execution_identity(config, row)
    row['fingerprint_inputs'] = fingerprint_inputs(row)
    row['reproducibility_fingerprint'] = digest(row['fingerprint_inputs'])
    return row


def reuse_identity(manifest, source, root):
    """Scientific invocation inputs, excluding checkpoint-local selection statistics."""
    return {'candidate_id': manifest['v2_derived']['selection']['candidate_id'], **_source_identity(source),
        'policy': manifest['execution_policy'], 'codex': manifest['codex'], 'dataset': manifest['dataset'],
        'evaluator': manifest['evaluator'], 'tasks': manifest['tasks']['ordered_ids'],
        'task_membership_hash': manifest['tasks']['task_membership_hash'],
        'task_order_hash': manifest['tasks']['task_order_hash'],
        'entry_hashes': manifest['v2_derived']['entry_hashes'],
        'model_fingerprint': manifest['evaluation_execution']['model_fingerprint'],
        'evaluator_fingerprint': manifest['evaluation_execution']['evaluator_fingerprint'], 'verified_root': root}


def inventory(directory):
    """Digest of every file (and link target) in one sealed final directory."""
    from run_metadata import file_hash
    rows = []
    for path in sorted(Path(directory).rglob('*')):
        if path.is_symlink():
            # Do not follow artifacts out of their sealed directory.
            rows.append([str(path.relative_to(directory)), 'link', str(path.readlink())])
        elif path.is_file():
            rows.append([str(path.relative_to(directory)), path.stat().st_size, file_hash(io._regular(path))])
    return io.digest(rows)


def implementation(cp):
    """Same-commit integrity proof of the frozen checkpoint before any derived write or callback."""
    from run_metadata import ROOT, repository_snapshot, check_repository
    from checkpoint_proof import state_proof
    repository = repository_snapshot(ROOT)
    check_repository(repository)
    old = cp.manifest['repository']['commit']
    check(repository['commit'] == old, 'BAI_IMPLEMENTATION_COMMIT')
    _, cached, _ = common_root(cp.manifest)
    proof = state_proof(cp, ExecutionRun(cp.directory, cp.manifest), cached)
    return {'old_commit': old, 'new_commit': old, 'checkpoint_digest': cp.digest, **proof}


def reuse_evidence(cp, target, source, source_budget, domain):
    check(type(source_budget) is int and source_budget <= cp.b and domain in ('primary', 'secondary', FINAL)
          and (source_budget < cp.b or domain != FINAL), 'BAI_REUSE_SCOPE')
    old = Checkpoint(cp.directory, source_budget)
    folder = old.output / domain
    check(not (folder / 'reuse.json').exists() and not (folder / 'final_reference.json').exists(),
          'BAI_REUSE_DIRECT_REQUIRED')
    manifest = io._read_json(folder / 'run_manifest.json')
    old_source = CheckpointSources(old, folder).get(manifest['v2_derived']['selection']['candidate_id'], read_only=True)
    root, _, _ = common_root(cp.manifest)
    old_root, _, _ = common_root(old.manifest)
    identity = reuse_identity(target, source, root)
    check(identity == reuse_identity(manifest, old_source, old_root), 'BAI_REUSE_IDENTITY')
    report = final_report(cp.directory, source_budget) if domain == FINAL else wgm_report(cp.directory, source_budget, domain)
    count = len(cp.final_tasks)
    check(report['complete'] and report['completed'] == count and not report['ambiguous'] and not report['error_ids']
          and len(report['evidence']) == count, 'BAI_REUSE_INCOMPLETE')
    # Every source plan/binding/artifact was verified by its own report reader.
    check(identity == reuse_identity(manifest, report, report['common_root']), 'BAI_REUSE_IDENTITY')
    body = {'schema_version': 1, 'checkpoint': cp.b, 'checkpoint_digest': cp.digest,
        'target_fingerprint': target['reproducibility_fingerprint'], 'source_checkpoint': source_budget,
        'source_domain': domain, 'source_inventory_digest': inventory(folder),
        'source_artifacts_digest': report['final_artifacts_digest'], 'execution_identity': identity}
    return body, report


def final_report(directory, budget, *, entries=None):
    from validated_runs import _entry_map
    cp = Checkpoint(directory, budget)
    config = config_from_manifest(cp.manifest)
    entries = entries if entries is not None else _entry_map(config, cp.pool)
    bai = read_bai(cp, entries=entries, require_complete=True)
    folder = cp.output / FINAL
    if not folder.exists():
        return None
    manifest = io._read_json(folder / 'run_manifest.json')
    expected = final_manifest(cp, config, entries, bai)
    check(manifest['fingerprint_inputs'] == fingerprint_inputs(expected), 'BAI_FINAL_MANIFEST_CHANGED')
    run = ExecutionRun(folder, manifest)
    selected = bai['selection']
    source = _source_identity(CheckpointSources(cp, folder).get(selected['candidate_id'], read_only=True))
    root, _, _ = common_root(cp.manifest)
    reference = folder / 'reuse.json'
    if reference.exists():
        saved = io._read_json(reference)
        check(saved['sha256'] == digest(saved['reference']) and not run.plans.read()
              and not run.ledger.evaluations(), 'BAI_REUSE_CHANGED')
        ref = saved['reference']
        body, original = reuse_evidence(cp, manifest, source, ref['source_checkpoint'], ref['source_domain'])
        check(body == ref, 'BAI_REUSE_CHANGED')
        result = deepcopy(original)
        result.pop('final_artifacts_digest')
        result.update(final_physical_model_calls=0, final_physical_evaluator_calls=0, submitted=0,
            configured_workers=0, effective_workers=0, observed_workers=0, worker_history=[],
            batch_wall_seconds=0., final_wall_seconds=0., task_duration_sum_seconds=0.,
            reused_checkpoint_final=True, reused_from_checkpoint=ref['source_checkpoint'],
            reused_from_domain=ref['source_domain'], reference_digest=saved['sha256'],
            source_final_physical_model_calls=original['final_physical_model_calls'],
            source_final_physical_evaluator_calls=original['final_physical_evaluator_calls'],
            source_final_artifacts_digest=original['final_artifacts_digest'])
    else:
        result = _binding_report(folder, cp.final_tasks, selected['candidate_id'], digest(manifest['v2_derived']), source)
        if (folder / 'derived_state.json').exists():
            engine = DerivedEvaluation(cp, run, entries, ReadOnlySources(cp, folder), None)
            state = engine.load()
            check(len(state['records']) <= result['completed'], 'BAI_FINAL_STATE_CHANGED')
    cp.unchanged()
    return _seal_report({**result, **source, 'algorithm': 'wgm_v2', 'profile': 'wgm_v2', 'role': 'bai_final',
        'checkpoint': budget, 'valid_search_budget': budget, 'canonical_budget': budget,
        'checkpoint_digest': cp.digest, 'candidate_id': selected['candidate_id'], 'selection': selected,
        'selection_digest': digest(manifest['v2_derived']), 'final_selection_source': 'checkpoint_bai',
        'final_selection_rule': RULE, 'bai_selection_digest': bai['selection_digest'],
        'extra_bai_evaluations': bai['completed'], 'total_selection_evaluations': budget + bai['completed'],
        'checkpoint_bai_calls': {k: bai[k] for k in ('physical_model_calls', 'physical_evaluator_calls')},
        'checkpoint_bai_wall_seconds': bai['wall_seconds'], 'common_root': root,
        'task_membership_hash': root['task_membership_hash'], 'task_order_hash': root['task_order_hash'],
        'execution_policy': cp.manifest['execution_policy']['effective_policy'],
        'search': search_accounting(cp), 'formal_comparison_eligible': _formal(cp),
        'derived_fingerprint': manifest['reproducibility_fingerprint']})


def report_bundle(directory, budget):
    from final_adapter import final_report as canonical_report
    cp = Checkpoint(directory, budget)
    bai = read_bai(cp)
    canonical = canonical_report(directory, budget)
    final = final_report(directory, budget) if (cp.output / FINAL).exists() else None
    return {**canonical, 'canonical_primary': cp.selection['primary'],
        'canonical_primary_final': canonical['primary'], 'checkpoint_bai_status': bai['status'],
        'checkpoint_bai_budget': bai['budget'], 'checkpoint_bai_selected_candidate': (
            bai['selection']['candidate_id'] if bai['selection'] else None),
        'checkpoint_bai_selection_digest': bai['selection_digest'], 'checkpoint_bai_final': final,
        'canonical_budget': budget, 'extra_bai_evaluations': bai['completed'],
        'total_selection_evaluations': budget + bai['completed'], 'checkpoint_bai': bai}


def run(directory, budget, mode, *, workers=None, preflight, retry_preparation=None, readiness=None):
    """Execution only through an explicit CLI invocation; completed BAI stays read-only."""
    from validated_runs import _entry_map, executor
    from run_metadata import RunRecorder
    check(mode in ('bai', 'final'), 'BAI_MODE')
    check(retry_preparation is None or mode == 'final', 'BAI_RETRY_FINAL_ONLY')
    cp = Checkpoint(directory, budget)
    proof = implementation(cp)
    config = config_from_manifest(cp.manifest, workers)
    if readiness is not None:
        config.network_readiness = config.codex.network_readiness = readiness
    entries = _entry_map(config, cp.pool)
    bai = read_bai(cp, entries=entries, require_complete=mode == 'final')
    if mode == 'bai' and bai['status'] == 'completed':
        return report_bundle(directory, budget)
    manifest = (make_manifest(cp, config, 'bai', entries) if mode == 'bai'
                else final_manifest(cp, config, entries, bai))
    folder = cp.output / ('bai' if mode == 'bai' else FINAL)
    folder.parent.mkdir(parents=True, exist_ok=True)
    if folder.exists():
        saved = io._read_json(folder / 'run_manifest.json')
        check(saved['reproducibility_fingerprint'] == manifest['reproducibility_fingerprint'], 'BAI_RESUME_IDENTITY')
        execution = ExecutionRun(folder, saved)
    else:
        execution = ExecutionRun.create(folder, manifest)
    with execution.lock():
        receipt = {'implementation': proof, 'derived_fingerprint': manifest['reproducibility_fingerprint']}
        path = folder / 'implementation.json'
        if path.exists():
            check(io._read_json(path) == receipt, 'BAI_IMPLEMENTATION_CHANGED')
        else:
            durable_json(path, receipt)
        if retry_preparation:
            from preparation_recovery import authorize
            authorize(execution, retry_preparation, proof)
        execution.precheck()
        if mode == 'final' and (folder / 'reuse.json').exists():
            return report_bundle(directory, budget)
        sources = CheckpointSources(cp, folder)
        candidates = ([r['commit_id'] for r in cp.state['nodes']] if mode == 'bai' else [bai['selection']['candidate_id']])
        for candidate in candidates:
            sources.get(candidate)
        if mode == 'final' and not execution.plans.read():
            # Deterministic direct-evidence search; references never form chains.
            for old in sorted((cp.directory / 'derived').glob('checkpoint-*')):
                old_budget = int(old.name.removeprefix('checkpoint-'))
                if old_budget > budget:
                    continue
                for domain in (FINAL, 'primary', 'secondary'):
                    other = old / domain
                    if other == folder or not (other / 'run_manifest.json').exists() or any(
                            (other / n).exists() for n in ('reuse.json', 'final_reference.json')):
                        continue
                    old_manifest = io._read_json(other / 'run_manifest.json')
                    if old_manifest['v2_derived']['selection']['candidate_id'] != candidates[0]:
                        continue
                    summary = other / 'summary.json'
                    if not summary.exists() or not io._read_json(summary).get('complete'):
                        continue
                    body, _ = reuse_evidence(cp, manifest, sources.get(candidates[0]), old_budget, domain)
                    durable_json(folder / 'reuse.json', {'reference': body, 'sha256': digest(body)})
                    result = final_report(directory, budget, entries=entries)
                    durable_json(folder / 'summary.json', result)
                    durable_json(folder / 'report.json', result)
                    return report_bundle(directory, budget)
        engine = DerivedEvaluation(cp, execution, entries, sources, None)
        if mode == 'final' and engine.path.exists() and engine.decision(engine.load())[1] is not None:
            return report_bundle(directory, budget)
        preflight(config, output_root=folder)
        recorder = RunRecorder(folder, execution.manifest, network_readiness=config.network_readiness)
        cancel = threading.Event()
        engine.execute = executor(config, execution, entries, sources, recorder, cancel_event=cancel)
        try:
            engine.run_to_completion(locked=True, workers=config.execution.evaluation_workers, cancel_event=cancel)
        except BaseException:
            recorder.refresh('incomplete')
            raise
        else:
            recorder.refresh('completed')
        finally:
            cp.unchanged()
        if mode == 'final':
            result = final_report(directory, budget, entries=entries)
            durable_json(folder / 'report.json', result, replace_existing=True)
    return report_bundle(directory, budget)
