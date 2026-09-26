"""Small, standard-library, single-run checkpoint efficiency report."""
from task_schedule import count_for_membership
import csv
import io as streams
import math
from itertools import combinations
from pathlib import Path

import evaluation_ledger as io
from evaluation_plan import check, durable_json
from comparison_report import comparison, report_text, wgm_selected
from final_adapter import final_report

CAUTION = ('Single run per algorithm; descriptive observed checkpoints only. Search and final tasks overlap. '
           'Intervals/p-values do not establish algorithmic significance or a general 2x improvement. '
           'Checkpoint B excludes extra checkpoint BAI: use total_selection_evaluations for selection cost. '
           'Benchmark sample efficiency, search wall time and final worker speed are different quantities.')


def wilson(successes, total, z=1.959963984540054):
    check(type(successes) is int and type(total) is int and 0 <= successes <= total and total > 0,
          'CURVE_COUNTS')
    p, z2 = successes / total, z*z
    center = (p + z2/(2*total)) / (1 + z2/total)
    half = z * math.sqrt(p*(1-p)/total + z2/(4*total*total)) / (1 + z2/total)
    return [max(0., center-half), min(1., center+half)]


def paired(left, right):
    left, right = set(left), set(right)
    a, b = len(left-right), len(right-left)
    n = a+b
    return {'left_exclusive': a, 'right_exclusive': b,
            'left_exclusive_ids': sorted(left-right), 'right_exclusive_ids': sorted(right-left),
            'intersection_ids': sorted(left & right), 'union_ids': sorted(left | right),
            'intersection': len(left & right), 'union': len(left | right),
            'jaccard': len(left & right)/len(left | right) if left | right else 1.,
            'exact_mcnemar_two_sided_p': min(1., 2*sum(math.comb(n, k) for k in range(min(a,b)+1))/2**n)}


def derive(rows):
    observed = [r for r in rows if r['status'] == 'complete']
    thresholds = sorted({r['accuracy'] for r in observed})
    minima = []
    for algorithm in sorted({r['algorithm'] for r in rows}):
        for accuracy in thresholds:
            matches = [r for r in observed if r['algorithm'] == algorithm and r['accuracy'] >= accuracy]
            best = min(matches, key=lambda r:r['checkpoint']) if matches else None
            minima.append({'algorithm': algorithm, 'observed_accuracy_threshold': accuracy,
                           'minimum_observed_checkpoint': best['checkpoint'] if best else None,
                           'total_selection_evaluations': best['total_selection_evaluations'] if best else None,
                           'search_wall_seconds_at_checkpoint': best['search_cumulative_wall_seconds'] if best else None})
    ratios = []
    for left, right in combinations(minima, 2):
        if left['observed_accuracy_threshold'] != right['observed_accuracy_threshold']:
            continue
        lb, rb = left['minimum_observed_checkpoint'], right['minimum_observed_checkpoint']
        lw, rw = left['search_wall_seconds_at_checkpoint'], right['search_wall_seconds_at_checkpoint']
        ls, rs = left['total_selection_evaluations'], right['total_selection_evaluations']
        ratios.append({'left':left['algorithm'], 'right':right['algorithm'],
                       'accuracy_threshold':left['observed_accuracy_threshold'],
                       'checkpoint_budget_ratio_left_over_right': lb/rb if lb and rb else None,
                       'total_selection_evaluation_ratio_left_over_right': ls/rs if ls and rs else None,
                       'wall_time_ratio_left_over_right': lw/rw if lw is not None and rw else None})
    frontiers = {}
    for cost in ('checkpoint', 'total_selection_evaluations', 'search_cumulative_wall_seconds'):
        eligible = [r for r in observed if r[cost] is not None]
        frontiers[cost] = [{'algorithm':r['algorithm'], 'checkpoint':r['checkpoint']} for r in eligible
                          if not any(s[cost] <= r[cost] and s['accuracy'] >= r['accuracy']
                                     and (s[cost] < r[cost] or s['accuracy'] > r['accuracy']) for s in eligible)]
    pairs = []
    for left, right in combinations(observed, 2):
        if left['algorithm'] != right['algorithm']:
            pairs.append({'left':[left['algorithm'],left['checkpoint']],
                          'right':[right['algorithm'],right['checkpoint']],
                          **paired(left['resolved_ids'], right['resolved_ids'])})
    return {'minimum_observed_checkpoint':minima, 'ratios':ratios, 'pareto_frontier':frontiers, 'paired':pairs}


def build(reports, metadata, checkpoints=(100, 200, 400, 800), *, wgm_result='bai'):
    """Pure computation. reports contains already independently verified finals."""
    rows, joints = [], {}
    common = [b for b,r in reports.items() if set(r) == {'hgm','dgm','wgm_v2'}
              and (wgm_selected(r['wgm_v2'], wgm_result) or {}).get('complete')]
    for b in sorted(common):
        joints[b] = comparison(reports[b], b, highest_common_budget=max(common), wgm_result=wgm_result)
    for algorithm in ('hgm','dgm','wgm_v2'):
        previous_wall, previous_budget = 0., 0
        for b in checkpoints:
            r = reports.get(b, {}).get(algorithm)
            r = wgm_selected(r, wgm_result) if r and algorithm == 'wgm_v2' else r
            if r is None or not r['complete']:
                rows.append({'algorithm':algorithm, 'checkpoint':b, 'status':'not_observed',
                    **dict.fromkeys(('candidate_id','selection_digest','resolved','accuracy','resolved_ids',
                       'search_cumulative_wall_seconds','incremental_wall_seconds','incremental_from_checkpoint',
                       'valid_benchmark_evaluations','stage_calls','stage_durations','readiness_wait_seconds',
                       'failed','ambiguous','scientific_commit','implementation_commit','wilson_95_interval',
                       'absolute_root_improvement','workers'))})
                continue
            # Even partial checkpoint collections must pass strict final validation.
            check(r['complete'] and r['checkpoint'] == r['valid_search_budget'] == b
                  and r['requested'] == r['completed'] == count_for_membership(r['task_membership_hash'])
                  and not r['error_ids'] and not r['ambiguous']
                  and r['algorithm'] == algorithm and r['profile'] == ('wgm_v2' if algorithm == 'wgm_v2' else 'legacy_code')
                  and r['formal_comparison_eligible'] and r['execution_policy'] == 'validated_observation_v2'
                  and all(r[k] == r['common_root'][k] for k in ('task_membership_hash','task_order_hash'))
                  and r['final_artifacts_digest'] == io.digest({k:v for k,v in r.items() if k != 'final_artifacts_digest'}),
                  'CURVE_FINAL_INVALID')
            s, wall = r['search'], r['search']['search_wall_seconds']
            check(wall is None or previous_wall is None or wall >= previous_wall, 'CURVE_WALL_DECREASE')
            extra = metadata[algorithm].get(str(b), {})
            row = {'algorithm':algorithm, 'checkpoint':b, 'status':'complete',
                'selection_source': r['final_selection_source'],
                'canonical_budget': b, 'extra_bai_evaluations': r.get('extra_bai_evaluations', 0),
                'total_selection_evaluations': r.get('total_selection_evaluations', b),
                'checkpoint_bai_calls': r.get('checkpoint_bai_calls'),
                'checkpoint_bai_wall_seconds': r.get('checkpoint_bai_wall_seconds'),
                'candidate_id':r['candidate_id'], 'selection_digest':r['selection_digest'],
                'source_digest':r.get('selection',{}).get('source_fingerprint', r.get('source_fingerprint')),
                'resolved':r['resolved'], 'accuracy':r['accuracy'], 'resolved_ids':r['resolved_ids'],
                'search_cumulative_wall_seconds':wall,
                'incremental_wall_seconds':wall-previous_wall if wall is not None and previous_wall is not None else None,
                'incremental_from_checkpoint':previous_budget, 'valid_benchmark_evaluations':b,
                'stage_calls':{k:s[k] for k in ('diagnosis_calls','generation_calls','downstream_model_calls','evaluator_calls')},
                'stage_durations':{**{k:s[k] for k in ('diagnosis_time_seconds','generation_time_seconds','downstream_evaluation_time_seconds')},
                                   'downstream_model_seconds':extra.get('downstream_model_seconds'),
                                   'evaluator_seconds':extra.get('evaluator_seconds')},
                'readiness_wait_seconds':extra.get('readiness_wait_seconds'),
                'failed':0, 'ambiguous':s['ambiguous_evaluations'],
                'scientific_commit':metadata[algorithm]['scientific_commit'],
                'implementation_commit':extra.get('implementation_commit'),
                'wilson_95_interval':wilson(r['resolved'],r['requested']),
                'absolute_root_improvement':r['accuracy']-r['common_root']['accuracy'],
                'workers':{'search':s['workers'], 'final':{k:r[k] for k in (
                    'configured_workers','effective_workers','observed_workers')}},
                'final_wall_seconds':r['final_wall_seconds'], 'final_artifacts_digest':r['final_artifacts_digest']}
            rows.append(row)
            if b in joints:
                joints[b]['algorithms'][algorithm]['search_incremental_wall_seconds'] = row['incremental_wall_seconds']
                joints[b]['algorithms'][algorithm]['search_incremental_from_checkpoint'] = previous_budget
            previous_wall, previous_budget = wall, b
    roots = {io.digest(r['common_root']) for group in reports.values() for a,r in group.items()
             for r in ([wgm_selected(r, wgm_result)] if a == 'wgm_v2' else [r]) if r is not None}
    check(len(roots) <= 1, 'CURVE_ROOT_IDENTITY')
    return {'schema_version':2, 'single_run':True, 'caution':CAUTION, 'rows':rows,
            'wgm_result': 'checkpoint_bai' if wgm_result == 'bai' else 'canonical_primary_ablation',
            'wgm_canonical_primary_ablation': {str(b):r['wgm_v2'].get('primary')
                                             for b,r in reports.items() if 'wgm_v2' in r},
            'statistics_definition':'Wilson 95% marginal intervals; exact two-sided conditional binomial on paired discordances; no multiplicity adjustment',
            'missing_evidence':'null, including historical readiness timing not frozen at checkpoint',
            'joint_reports':joints, **derive(rows)}


def text_report(result):
    import json
    lines = [CAUTION, 'WGM result: ' + result['wgm_result'],
             'algorithm B resolved accuracy search_seconds incremental_seconds']
    for r in result['rows']:
        lines.append(f"{r['algorithm']} {r['checkpoint']} {r['resolved']} {r['accuracy']} "
                     f"{r['search_cumulative_wall_seconds']} {r['incremental_wall_seconds']}")
        lines.append('  ' + json.dumps({k:r.get(k) for k in (
            'status','candidate_id','selection_digest','selection_source','canonical_budget',
            'extra_bai_evaluations','total_selection_evaluations','checkpoint_bai_calls',
            'checkpoint_bai_wall_seconds','valid_benchmark_evaluations','stage_calls',
            'stage_durations','readiness_wait_seconds','failed','ambiguous','scientific_commit',
            'implementation_commit','wilson_95_interval','absolute_root_improvement','workers')}, sort_keys=True))
    for key in ('minimum_observed_checkpoint','ratios','pareto_frontier','paired'):
        lines.append(key + ': ' + json.dumps(result[key], sort_keys=True))
    return '\n'.join(lines)+'\n'


def write_reports(result, output):
    from candidate_workspace import _atomic_write
    import json
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    durable_json(output/'comparison-curve.json', result, replace_existing=True)
    _atomic_write(output/'comparison-curve.txt', text_report(result).encode())
    stream = streams.StringIO()
    fields = sorted(set().union(*(r.keys() for r in result['rows'])))
    writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
    for row in result['rows']:
        writer.writerow({k:json.dumps(v, sort_keys=True) if isinstance(v,(list,dict)) else v for k,v in row.items()})
    _atomic_write(output/'comparison-curve.csv', stream.getvalue().encode())


def curve(directories, output, *, wgm_result='bai'):
    from comparison_root import DEFAULT_ROOTS
    output = Path(output).resolve()
    roots = {k:Path(v).resolve() for k,v in directories.items()}
    check(all(not output.is_relative_to(p) and not p.is_relative_to(output)
              for p in (*roots.values(), *DEFAULT_ROOTS.values())),
          'COMPARE_OUTPUT_INSIDE_RUN')
    reports, metadata = {}, {}
    for algorithm, directory in roots.items():
        manifest = io._read_json(directory/'run_manifest.json')
        metadata[algorithm] = {'scientific_commit':manifest['repository']['commit']}
        for b in (100,200,400,800):
            if not (directory/'derived'/f'checkpoint-{b:06d}').exists():
                continue
            if algorithm == 'wgm_v2':
                from wgm_bai_final import report_bundle
                row = report_bundle(directory, b)
            else:
                row = final_report(directory,b)
            reports.setdefault(b,{})[algorithm] = row
            # Runs resume only on their own commit, so the manifest commit is the
            # execution commit. A report reader's current HEAD is never used.
            metadata[algorithm][str(b)] = {'implementation_commit':manifest['repository']['commit']}
            metadata[algorithm][str(b)].update(stage_durations(directory,b,manifest))
            summary = io._read_json(directory/'run_summary.json')
            current = io._read_json(directory/'search_state.json')['state']
            if current.get('valid_evaluations',current.get('n_task_evals')) == b:
                readiness = summary.get('network_readiness',{})
                metadata[algorithm][str(b)]['readiness_wait_seconds'] = readiness.get('total_wait_seconds')
    result = build(reports, metadata, wgm_result=wgm_result)
    write_reports(result,output)
    if 200 in result['joint_reports']:
        joint = result['joint_reports'][200]
        durable_json(output/'comparison.json', joint, replace_existing=True)
        from candidate_workspace import _atomic_write
        _atomic_write(output/'comparison.txt', (report_text(joint)+'\n'+CAUTION+'\n').encode())
    return result


def stage_durations(directory, budget, manifest):
    """Keep missing/waived process duration unknown; never substitute zero."""
    from evaluation_scheduler import ExecutionRun
    run = ExecutionRun(directory,manifest)
    state = io._read_json(directory/'checkpoints'/f'{budget:06d}.json')['state']
    consumed = state.get('execution',state)['consumed_bindings']
    values, markers = {'model':[], 'evaluation':[]}, {'model':0, 'evaluation':0}
    for snapshot in run.ledger.evaluations():
        if snapshot.identity.evaluation_id not in consumed:
            continue
        for event in snapshot.events:
            for stage in values:
                if event.kind == stage+'_started':
                    markers[stage] += 1
                elif event.kind == stage+'_terminal':
                    values[stage].append(io._decode(event.data).get('evidence',{}).get('duration_seconds'))
    return {name:sum(values[stage]) if len(values[stage]) == markers[stage] and all(
        type(t) in (int,float) and math.isfinite(t) and t >= 0 for t in values[stage]) else None
        for stage,name in (('model','downstream_model_seconds'),('evaluation','evaluator_seconds'))}
