"""Read-only validation and joint reporting of three completed checkpoint finals."""
from itertools import combinations
from pathlib import Path

from evaluation_plan import check, durable_json
from final_adapter import final_report
from run_metadata import digest


def wgm_selected(row, result):
    check(result in ('bai', 'canonical'), 'COMPARE_WGM_RESULT')
    selected = row.get('checkpoint_bai_final') if result == 'bai' else row.get('primary')
    if selected is not None:
        check(selected['final_selection_source'] == ('checkpoint_bai' if result == 'bai' else 'wgm_v2_checkpoint'),
              'COMPARE_WGM_SELECTION_SOURCE')
    return selected


def read_results(directories, b):
    from wgm_bai_final import report_bundle
    return {k: report_bundle(p, b) if k == 'wgm_v2' else final_report(p, b) for k,p in directories.items()}


def comparison(rows, b, *, highest_common_budget, wgm_result='bai'):
    check(set(rows) == {'hgm', 'dgm', 'wgm_v2'}, 'COMPARE_PROFILES')
    wgm = rows['wgm_v2']
    selected = {**rows, 'wgm_v2': wgm_selected(wgm, wgm_result)}
    check(all(r is not None for r in selected.values()), 'COMPARE_FINAL_INCOMPLETE')
    root = selected['hgm']['common_root']
    for algorithm, row in selected.items():
        check(row['algorithm'] == algorithm and row['profile'] == ('wgm_v2' if algorithm == 'wgm_v2' else 'legacy_code'),
              'COMPARE_PROFILES')
        check(row['checkpoint'] == row['valid_search_budget'] == b, 'COMPARE_CHECKPOINT')
        check(row['common_root'] == root, 'COMPARE_ROOT_IDENTITY')
        check(row['task_membership_hash'] == root['task_membership_hash'] and
              row['task_order_hash'] == root['task_order_hash'], 'COMPARE_TASK_IDENTITY')
        check(row['execution_policy'] == 'validated_observation_v2', 'COMPARE_EXECUTION_POLICY')
        from task_schedule import count_for_membership
        check(row['complete'] and row['completed'] == row['requested'] == count_for_membership(row['task_membership_hash'])
              and not row['ambiguous']
              and not row['error_ids'] and row['formal_comparison_eligible'], 'COMPARE_FINAL_INCOMPLETE_OR_SMOKE')
        check(row['final_artifacts_digest'] == digest({k:v for k,v in row.items() if k != 'final_artifacts_digest'}),
              'COMPARE_ARTIFACTS_CHANGED')
    output = {}
    for algorithm, row in selected.items():
        search_wall, final_wall = row['search']['search_wall_seconds'], row['final_wall_seconds']
        output[algorithm] = {**row, 'absolute_improvement': row['accuracy'] - root['accuracy'],
            'improvement_percentage_points': 100 * (row['accuracy'] - root['accuracy']),
            'total_wall_seconds': None if search_wall is None or final_wall is None else search_wall + final_wall}
        if algorithm == 'wgm_v2' and wgm_result == 'bai':
            bai_wall = row['checkpoint_bai_wall_seconds']
            output[algorithm]['total_wall_seconds'] = (None if None in (search_wall, final_wall, bai_wall)
                                                       else search_wall + bai_wall + final_wall)
    pairs = {}
    for a,bname in combinations(selected, 2):
        left, right = set(selected[a]['resolved_ids']), set(selected[bname]['resolved_ids'])
        pairs[a + '_vs_' + bname] = {'left_exclusive': len(left - right), 'right_exclusive': len(right - left),
            'left_exclusive_ids': sorted(left - right), 'right_exclusive_ids': sorted(right - left),
            'intersection_ids': sorted(left & right), 'union_ids': sorted(left | right),
            'intersection': len(left & right), 'union': len(left | right),
            'jaccard': len(left & right) / len(left | right) if left | right else 1.0}
    secondary = wgm.get('secondary')
    return {'schema_version': 2, 'checkpoint': b, 'highest_common_budget': highest_common_budget,
        'wgm_result': 'checkpoint_bai' if wgm_result == 'bai' else 'canonical_primary_ablation',
        'wgm_canonical_primary_ablation': wgm.get('canonical_primary_final', wgm.get('primary')),
        'is_highest_common_budget': b == highest_common_budget,
        'root_accuracy': root['accuracy'], 'common_root': root, 'algorithms': output, 'pairwise': pairs,
        'wgm_secondary': secondary, 'wgm_primary_secondary': (
            None if secondary is None or wgm.get('primary') is None else _secondary_metrics(wgm['primary'], secondary)),
        'task_level_held_out': False, 'wall_time_definition': 'search + extra checkpoint BAI (main WGM) + final; unavailable timing=null'}


def _secondary_metrics(primary, secondary):
    if not secondary['complete']:
        return None
    left, right = set(primary['resolved_ids']), set(secondary['resolved_ids'])
    return {'primary_exclusive': len(left-right), 'secondary_exclusive': len(right-left),
            'intersection': len(left & right), 'union': len(left | right),
            'jaccard': len(left & right)/len(left | right) if left | right else 1.0}


def report_text(row):
    lines = [f"Checkpoint {row['checkpoint']} (highest common completed budget: {row['highest_common_budget']})",
             f"WGM result: {row['wgm_result']}; canonical Primary retained separately as ablation",
             f"Root: {row['common_root']['resolved']}/{row['common_root'].get('completed', row['algorithms'][next(iter(row['algorithms']))]['requested'])} ({row['root_accuracy']:.2%})"]
    for name, r in row['algorithms'].items():
        s = r['search']
        lines += [f"{name}: {r['resolved']}/{r['requested']} ({r['accuracy']:.2%}), improvement={r['improvement_percentage_points']:+.2f} percentage points",
            f"  selected={r['candidate_id']}; rule={r['final_selection_rule']}",
            f"  canonical budget={r['checkpoint']}; extra BAI={r.get('extra_bai_evaluations', 0)}; total selection evaluations={r.get('total_selection_evaluations', r['checkpoint'])}",
            f"  search evaluations={s['valid_search_evaluations']}; diagnosis/generation/downstream/evaluator="
            f"{s['diagnosis_calls']}/{s['generation_calls']}/{s['downstream_model_calls']}/{s['evaluator_calls']}",
            f"  wall seconds search/final/total={s['search_wall_seconds']}/{r['final_wall_seconds']}/{r['total_wall_seconds']}",
            f"  incremental search seconds={r.get('search_incremental_wall_seconds')} from B={r.get('search_incremental_from_checkpoint')}",
            f"  final workers configured/effective/observed={r['configured_workers']}/{r['effective_workers']}/{r['observed_workers']}",
            f"  final calls model/evaluator={r['final_physical_model_calls']}/{r['final_physical_evaluator_calls']}; reused root={r['reused_verified_root']}",
            f"  search schedule={s['search_schedule_hash']}; resolved IDs=" + ','.join(r['resolved_ids'])]
    for names, p in row['pairwise'].items():
        lines.append(f"{names}: exclusive={p['left_exclusive']}/{p['right_exclusive']}, intersection={p['intersection']}, union={p['union']}, Jaccard={p['jaccard']:.6f}")
    lines.append('WGM Secondary: ' + ('not executed (null)' if row['wgm_secondary'] is None else row['wgm_secondary']['status']))
    return '\n'.join(lines)


def compare(directories, b, output, *, wgm_result='bai'):
    roots = {k:Path(v).resolve() for k,v in directories.items()}
    output = Path(output).resolve()
    from comparison_root import DEFAULT_ROOTS
    check(all(not output.is_relative_to(p) and not p.is_relative_to(output) for p in roots.values()) and
          not any(output.is_relative_to(p) for p in DEFAULT_ROOTS.values()), 'COMPARE_OUTPUT_INSIDE_RUN')
    rows = read_results(roots, b)
    # Only complete final evidence qualifies as a common budget; filenames are not proof.
    available = []
    for p in roots.values():
        available.append({int(q.name.removeprefix('checkpoint-')) for q in (p / 'derived').glob('checkpoint-*')
                          if q.name.removeprefix('checkpoint-').isdigit()})
    common = set.intersection(*available)
    highest = b
    for budget in sorted(x for x in common if x > b):
        try:
            other = read_results(roots, budget)
            comparison(other, budget, highest_common_budget=budget, wgm_result=wgm_result)
        except (ValueError, OSError, KeyError):
            continue
        highest = budget
    result = comparison(rows, b, highest_common_budget=highest, wgm_result=wgm_result)
    output.mkdir(parents=True, exist_ok=True)
    durable_json(output / 'comparison.json', result, replace_existing=True)
    from candidate_workspace import _atomic_write
    _atomic_write(output / 'comparison.txt', (report_text(result) + '\n').encode())
    return result
