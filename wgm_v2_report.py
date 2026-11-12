"""Read-only canonical V2 reports; no scheduler lease, execution or reconciliation.

Validated bindings are the outcome evidence. Summaries are checked views, never
substitutes for journal verification. Reporting does not restore live RNG/tree
objects or materialize source workspaces. Derived evaluations remain separate.
"""
from collections import Counter
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace

import evaluation_ledger as io
from candidate_provenance import read_chain
from evaluation_plan import check
from evaluation_scheduler import ExecutionRun, read_binding, resume_decision
from search_operation_journal import SearchOperationJournal, classify
from search_operations import read_checkpoint
from wgm_v2_derived import accounting, snapshot_selection
from wgm_v2_search import physical_accounting, smoke_coverage


def _duration(values, expected):
    if len(values) != expected or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values):
        return None
    return sum(values)


def read_search_report(directory, manifest):
    directory = Path(directory)
    run = ExecutionRun(directory, manifest)  # Verifies manifest identity/fingerprint, without a lease.
    checkpoint = read_checkpoint(directory)  # Includes state digest and structured RNG validation.
    state, control = checkpoint['state'], checkpoint['runtime_control']
    execution = state['execution']
    check(checkpoint['fingerprint'] == manifest['reproducibility_fingerprint']
          and state['experiment'] == manifest['experiment']
          and state['experiment']['algorithm'] == 'wgm_v2'
          and execution['run_id'] == manifest['run_id']
          and execution['run_fingerprint'] == checkpoint['fingerprint'], 'REPORT_SEARCH_IDENTITY')
    saved = io._read_json(directory / 'wgm_v2_summary.json')
    run_summary = io._read_json(directory / 'run_summary.json')
    check(all(saved.get(k) == v for k, v in state['summary'].items())
          and saved['runtime_control'] == control and saved['protocol'] == control['protocol'],
          'REPORT_SEARCH_SUMMARY_CHANGED')

    plans = {p.identity.evaluation_id: p for p in run.plans.read()}
    snapshots = {s.identity.evaluation_id: s for s in run.ledger.evaluations()}
    check(set(snapshots) <= set(plans), 'REPORT_UNCLAIMED_EVALUATION')
    consumed = execution['consumed_bindings']
    observations = execution['observations']
    check(set(consumed) == set(observations) and set(consumed) <= set(plans), 'REPORT_BINDING_IDENTITY')
    rows, outcomes = [], {}
    durations = {'model': [], 'evaluation': []}
    for eid, plan in plans.items():
        inspection = run.ledger.inspect(eid, plan.identity)
        check(not inspection.error_code and not inspection.read_only and not inspection.legacy,
              'REPORT_LEDGER_INVALID')
        decision = resume_decision(plan, inspection, consumed)
        check(not decision.startswith('BLOCK_BINDING'), 'REPORT_BINDING_DIGEST_CHANGED')
        result = read_binding(run.ledger, plan) if decision.startswith('BOUND_') else None
        if eid in consumed:
            check(result is not None and result.outcome == 'success'
                  and consumed[eid] == result.binding_digest
                  and type(observations[eid]) is int and observations[eid] in (0, 1)
                  and observations[eid] == int(result.resolved), 'REPORT_OBSERVATION_CHANGED')
            outcomes[eid] = result
        rows.append({'evaluation_id': eid, 'task_id': plan.identity.task_id,
                     'candidate_id': plan.identity.candidate_id, 'task_draw_index': plan.identity.task_draw_index,
                     'decision': decision, 'outcome': result.outcome if result else None,
                     'resolved': result.resolved if result else None, 'consumed': eid in consumed})
        if inspection.snapshot:
            for event in inspection.snapshot.events:
                for stage in durations:
                    if event.kind == stage + '_terminal':
                        durations[stage].append(io._decode(event.data).get('evidence', {}).get('duration_seconds'))
    rows.sort(key=lambda r: r['task_draw_index'])
    slots = [s for s in state['slots'].values() if s['status'] == 'completed']
    check(len(slots) == len(outcomes) == state['n_task_evals'] == saved['valid_search_evaluations']
          and {s['evaluation_id'] for s in slots} == set(outcomes), 'REPORT_BUDGET_CHANGED')
    for slot in slots:
        result = outcomes[slot['evaluation_id']]
        identity = result.plan.identity
        check(slot['binding_digest'] == result.binding_digest and slot['observation'] == int(result.resolved)
              and all(slot[k] == getattr(identity, k) for k in ('candidate_id', 'task_id', 'task_draw_index')),
              'REPORT_SLOT_CHANGED')
    for node in state['nodes']:
        own = sorted((s for s in slots if s['candidate_id'] == node['commit_id']), key=lambda s: s['task_draw_index'])
        check(node['utility_measures'] == [s['observation'] for s in own], 'REPORT_NODE_OBSERVATIONS_CHANGED')
    check(saved['task_draws'] == len(state['draws'])
          and saved['unique_tasks'] == len({d['task_id'] for d in state['draws']})
          and saved['repeat_draws'] == saved['task_draws'] - saved['unique_tasks'], 'REPORT_TASK_DRAWS_CHANGED')

    journal = SearchOperationJournal(directory, manifest['run_id'], checkpoint['fingerprint'])
    op_plans = journal.plans()
    committed = execution['consumed_operations']
    check(set(committed) <= {p.operation_id for p in op_plans}, 'REPORT_OPERATION_MISSING')
    op_decisions = Counter()
    for plan in op_plans:
        operation, _ = journal.inspect(plan)  # Hash chain, durable head and sealed artifact digests.
        op_decisions[classify(operation, committed.get(plan.operation_id))] += 1
        if plan.operation_id in committed and operation['terminal']['outcome'] == 'candidate':
            terminal = operation['terminal']
            chain = read_chain(directory, plan.identity['candidate_id'], require_versioned=True)
            artifact = journal.read_artifact(plan, terminal['artifact'])
            metadata = io._read_json(directory / plan.identity['candidate_id'] / 'metadata.json')
            check(io.digest(chain[-1].provenance.to_dict()) == terminal['candidate_digest']
                  and hashlib.sha256(chain[-1].patch_bytes).hexdigest() == artifact['patch_digest']
                  and {k: v for k, v in metadata.items() if k != 'overall_performance'} ==
                      {k: v for k, v in artifact['metadata'].items() if k != 'overall_performance'},
                  'REPORT_CANDIDATE_CHANGED')
    counts = physical_accounting(run, SimpleNamespace(journal=journal))
    check(all(saved.get(k) == v for k, v in counts.items()), 'REPORT_PHYSICAL_COUNTS_CHANGED')
    times = accounting(directory)
    counts['total_physical_calls'] = times['total_physical_calls']
    coverage = smoke_coverage(SimpleNamespace(
        initialized=state['initialized'], founders=state['founders'], history=state['history'],
        initial_attempts=state['initial_attempts'], draws=state['draws'], execution=execution,
        selector=SimpleNamespace(draws=state['selector']['draws']) if state['selector'] else None,
        config=SimpleNamespace(wgm_v2=SimpleNamespace(**state['experiment']['parameters']))))
    check(coverage == saved['smoke_coverage'], 'REPORT_COVERAGE_CHANGED')
    coverage.update(initialization_attempts=state['initial_attempts'],
                    bai_evaluations=saved['completed_by_domain']['bai'])
    selection = io._read_json(directory / 'wgm_v2_selection.json')
    check(selection['canonical_state_digest'] == checkpoint['sha256']
          and all(selection[k] == v for k, v in snapshot_selection(state).items()), 'REPORT_SELECTION_CHANGED')
    if 'runtime_fingerprint' in manifest:
        from run_metadata import runtime_fingerprint, context_usage
        check(runtime_fingerprint(manifest) == manifest['runtime_fingerprint']
              and manifest['diagnosis_context_usage'] == context_usage(
                  [io._read_json(p) for p in (directory / 'codex_calls').glob('*.json')]),
              'REPORT_RUNTIME_FINGERPRINT_CHANGED')
    pending = sum(r['decision'] != 'BOUND_CONSUMED' and r['outcome'] != 'failed' for r in rows)
    incomplete_operations = len(op_plans) - len(committed)
    # The search stage of total budget B ends at the first iteration boundary where a full
    # iteration (M child evaluations + BAI) no longer fits in B - B_fin (Algorithm 1).
    params, n = state['experiment']['parameters'], state['n_task_evals']
    limit = control['target_budget'] - math.floor(params['checkpoint_bai_fraction'] * control['target_budget'])
    stopped = state['round'] is None and (state['stop_reason'] is not None or (
        n + params['M'] + params['bai_evals_per_iteration'] > limit if state['initialized'] else n >= limit))
    complete = (stopped and not state['inflight']
                and not saved['error_code'] and not pending and not incomplete_operations)
    check(run_summary['status'] != 'completed' or complete, 'REPORT_COMPLETION_CHANGED')
    network_path = directory / 'network_readiness.json'
    network = io._read_json(network_path) if network_path.exists() else {}
    network = {k: network[k] for k in ('enabled', 'ever_enabled', 'probe_attempts', 'receipt_count', 'wait_cycles',
        'timeout_count', 'cancel_count', 'last_safe_category', 'total_wait_seconds') if k in network}
    # Inventory only: canonical reporting must not run the derived report writer.
    derived = {}
    for label, domain in (('checkpoint_bai', 'bai'), ('primary_final', 'primary'), ('secondary_final', 'secondary')):
        paths = sorted((directory / 'derived').glob(f'checkpoint-*/{domain}/summary.json'))
        derived[label] = {'status': 'available_separately' if paths else 'unavailable',
                          'reason': 'explicit_checkpoint_report' if paths else 'not_executed',
                          'summary_files': [str(p.relative_to(directory)) for p in paths]}
    return {
        'evaluation_parallelism': manifest.get('evaluation_parallelism'),
        'report_type': 'wgm_v2_canonical_search', 'schema_version': 1,
        'status': run_summary['status'], 'canonical_status': saved['status'], 'target_completed': complete,
        'protocol': control['protocol'], 'algorithm': 'wgm_v2', 'commit': manifest['repository']['commit'],
        'execution_policy': manifest['execution_policy']['effective_policy'], 'codex': manifest['codex'],
        'planning_horizon': state['experiment']['parameters']['planning_horizon'], 'target_budget': control['target_budget'],
        'valid_search_evaluations': len(outcomes), 'resolved': sum(r.resolved for r in outcomes.values()),
        'unresolved': sum(not r.resolved for r in outcomes.values()),
        'formal_comparison_eligible': saved['formal_comparison_eligible'], 'production_ready': False,
        'canonical_state_digest': checkpoint['sha256'], 'reproducibility_fingerprint': checkpoint['fingerprint'],
        'integrity': 'verified_manifest_checkpoint_rng_journals_bindings_artifacts_selection',
        'smoke_coverage': coverage, 'lineage_count': len(state['founders']),
        'task_draws': len(state['draws']), 'unique_tasks': saved['unique_tasks'], 'repeat_draws': saved['repeat_draws'],
        'evaluations': rows, 'calls': counts, 'operation_decisions': dict(op_decisions),
        'failures': {**{k: counts[k] for k in ('failed_evaluations', 'ambiguous_evaluations', 'infrastructure_operations',
                    'ambiguous_operations', 'model_output_invalid_attempts')},
                    'candidate_invalid': saved['invalid_expansions'], 'incomplete_evaluations': pending,
                    'incomplete_operations': incomplete_operations, 'error_code': saved['error_code']},
        'timing': {'canonical_search_wall_seconds': run_summary['timing'].get('whole_run_total'),
                   'diagnosis_seconds': times['diagnosis_time_seconds'], 'generation_seconds': times['generation_time_seconds'],
                   'downstream_model_seconds': _duration(durations['model'], counts['downstream_model_calls']),
                   'evaluator_seconds': _duration(durations['evaluation'], counts['physical_evaluator_calls']),
                   'downstream_evaluation_seconds': times['downstream_evaluation_time_seconds'],
                   'definition': times['time_definition']},
        'network_readiness': network, 'selection': selection, 'derived': derived,
        'task_level_held_out': False, 'D_design': None,
    }


def search_report_text(report):
    r, c, coverage, t = report, report['calls'], report['smoke_coverage'], report['timing']
    seconds = lambda value: 'unavailable' if value is None else f'{value:.3f}'
    lines = [
        f"WGM V2 canonical search: {r['status']} (canonical={r['canonical_status']}, target_completed={str(r['target_completed']).lower()})",
        f"Protocol: {r['protocol']}; algorithm={r['algorithm']}; formal_comparison_eligible={str(r['formal_comparison_eligible']).lower()}",
        f"Run commit: {r['commit']}",
        f"Policy: {r['execution_policy']}; model={r['codex']['configured_model']}; reasoning={r['codex']['configured_reasoning_effort']}",
        f"Budget: planning_horizon={r['planning_horizon']}; target={r['target_budget']}; valid={r['valid_search_evaluations']}",
        f"Search observations: resolved={r['resolved']}; unresolved={r['unresolved']} (search events, not final60 accuracy)",
        f"Coverage: initialization_complete={str(coverage['initialization_complete']).lower()}; initialization_attempts={coverage['initialization_attempts']}; lineages={r['lineage_count']} (includes root)",
        f"EXP3 selections={coverage['exp3_selections']}; parent selections={coverage['bai_parent_selections']}; BAI evaluations={coverage['bai_evaluations']}",
        f"N2 committed operations={coverage['operation_commits']}; validated bindings={coverage['validated_bindings']}; uncovered_reason={coverage['post_init_uncovered_reason']}",
        f"IID draws={r['task_draws']}; unique tasks={r['unique_tasks']}; repeats={r['repeat_draws']}",
        f"Calls: diagnosis={c['diagnosis_calls']}; generation={c['generation_calls']}; downstream={c['downstream_model_calls']}; evaluator={c['physical_evaluator_calls']}; total={c['total_physical_calls']} (process start confirmations)",
        'Failures: ' + '; '.join(f'{k}={v}' for k, v in r['failures'].items()),
        f"Seconds: wall={seconds(t['canonical_search_wall_seconds'])}; diagnosis={seconds(t['diagnosis_seconds'])}; generation={seconds(t['generation_seconds'])}; downstream_model={seconds(t['downstream_model_seconds'])}; evaluator={seconds(t['evaluator_seconds'])}",
        'Network readiness: ' + '; '.join(f'{k}={v}' for k, v in r['network_readiness'].items()),
    ]
    for role in ('primary', 'secondary'):
        selected = r['selection'][role]
        lines.append(f"{role.capitalize()}: " + (
            f"lineage={selected['lineage_id']}; candidate={selected['candidate_id']}; n={selected['n']}; successes={selected['successes']}; lower={selected['lower']}; chain={' -> '.join(selected['parent_chain'])}"
            if selected else 'unavailable; ' + str(r['selection'].get('secondary_reason'))))
    lines.append('Selection: maximum search lower bound; ties by candidate UTF-8 bytes within lineage, then ascending lineage ID')
    if r.get('evaluation_parallelism'):
        workers = r['evaluation_parallelism']
        lines.append(f"Evaluation workers: configured={workers['configured_workers']}; effective={workers['effective_workers']}; canonical search remains serial")
    lines.extend(f"{k}: {v['status']} ({v['reason']})" for k, v in r['derived'].items())
    lines.extend(['Final60 accuracy/diversity: unavailable; task_level_held_out=false; D_design=null',
                  f"Integrity: {r['integrity']}", f"Canonical state SHA256: {r['canonical_state_digest']}",
                  'Production ready: false; TRANSIENT_NETWORK_REPLACEMENT_NOT_IMPLEMENTED',
                  'Task outcomes (canonical draw order):'])
    lines.extend(f"  {row['task_id']}: " + ('resolved' if row['resolved'] else 'unresolved')
                 if row['resolved'] is not None else f"  {row['task_id']}: {row['decision']}" for row in r['evaluations'])
    return '\n'.join(lines)
