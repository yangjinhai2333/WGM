"""Read-only integrity proof of a frozen search checkpoint; no execution or RNG restoration."""
import evaluation_ledger as io
from evaluation_plan import check


def progress(before, after):
    """Immutable evidence and admission prefixes, with append-only observations."""
    check(before['experiment'] == after['experiment'] and before['tasks'] == after['tasks']
          and before['n_task_evals'] <= after['n_task_evals']
          and after['history'][:len(before['history'])] == before['history'], 'FINAL_CHECKPOINT_ANCESTRY')
    for key in ('consumed_bindings', 'observations', 'consumed_operations'):
        check(all(after['execution'][key].get(k) == v for k, v in before['execution'][key].items()),
              'FINAL_CHECKPOINT_ANCESTRY')
    check(len(after['nodes']) >= len(before['nodes']), 'FINAL_CHECKPOINT_ANCESTRY')
    for a, b in zip(before['nodes'], after['nodes']):
        check(all(a[k] == b[k] for k in ('id', 'commit_id', 'parent_id'))
              and b['utility_measures'][:len(a['utility_measures'])] == a['utility_measures'],
              'FINAL_CHECKPOINT_ANCESTRY')
    check(all(after['branches'].get(k) == v for k, v in before['branches'].items()),
          'FINAL_CHECKPOINT_ANCESTRY')
    if 'v2_schema' in before:
        check(after['draws'][:len(before['draws'])] == before['draws']
              and all(after['slots'].get(k) == v for k, v in before['slots'].items()
                      if v['status'] == 'completed')
              and all(after['admission'].get(k) == v for k, v in before['admission'].items())
              and before['founders'] == after['founders'], 'FINAL_CHECKPOINT_ANCESTRY')


def state_proof(cp, run, cached):
    """Check all consumed bindings, N2 terminals, candidate provenance and RNG schema."""
    from evaluation_scheduler import read_binding
    from search_operation_journal import SearchOperationJournal, validate_rng
    from search_operations import checkpoint_rng
    from candidate_provenance import read_chain
    state = cp.state
    validate_rng(checkpoint_rng(state))  # Pure: never sets Python/NumPy/selector RNG.
    check(state['inflight'] is None and state['tasks'] == run.manifest['tasks']['ordered_ids']
          and state['experiment'] == run.manifest['experiment'], 'FINAL_CHECKPOINT_STATE')
    context = state['execution']
    consumed = context['consumed_bindings']
    plans = run.plans.read()
    check(len(consumed) == state['n_task_evals'] and set(consumed) == set(context['observations'])
          and {p.identity.evaluation_id for p in plans[:len(consumed)]} == set(consumed),
          'FINAL_CHECKPOINT_BINDINGS')
    nodes = {n['commit_id']: n for n in state['nodes']}
    check(len(nodes) == len(state['nodes']) and [n['id'] for n in state['nodes']] == list(range(len(nodes)))
          and all(n['parent_id'] is None if i == 0 else type(n['parent_id']) is int and 0 <= n['parent_id'] < i
                  for i, n in enumerate(state['nodes'])), 'FINAL_CHECKPOINT_TREE')
    from wgm_v2_derived import CheckpointSources
    source_cp = cp.sources_checkpoint() if hasattr(cp, 'sources_checkpoint') else cp
    sources = CheckpointSources(source_cp, cp.directory)
    identities = {c: sources.get(c, read_only=True) for c in nodes}
    heads, own = [], {c: [] for c in nodes}
    for index, plan in enumerate(plans[:len(consumed)]):
        eid = plan.identity.evaluation_id
        result = read_binding(run.ledger, plan)
        inspection = run.ledger.inspect(eid, plan.identity)
        check(result.outcome == 'success' and result.binding_digest == consumed[eid]
              and context['observations'][eid] == int(result.resolved)
              and plan.identity.stage == 'search' and plan.identity.candidate_id in nodes,
              'FINAL_CHECKPOINT_BINDING')
        check(all(getattr(plan.identity, k) == identities[plan.identity.candidate_id][k]
                  for k in ('source_fingerprint', 'policy_fingerprint')), 'FINAL_CHECKPOINT_SOURCE')
        own[plan.identity.candidate_id].append(int(result.resolved))
        heads.append([eid, result.binding_digest, inspection.snapshot.events[-1].sha256])
        if 'v2_schema' in state:
            draw = state['draws'][index]
            slot = state['slots'][draw['reservation_id']]
            check(draw == {'ordinal': index, 'task_id': plan.identity.task_id,
                  'candidate_id': plan.identity.candidate_id, 'reservation_id': draw['reservation_id'],
                  'evaluation_id': eid} and slot['status'] == 'completed'
                  and slot['evaluation_id'] == eid and slot['binding_digest'] == result.binding_digest
                  and slot['observation'] == int(result.resolved)
                  and slot['task_draw_index'] == plan.identity.task_draw_index == index
                  and slot['task_id'] == plan.identity.task_id
                  and slot['candidate_id'] == plan.identity.candidate_id, 'FINAL_CHECKPOINT_DRAW')
    if 'v2_schema' in state:
        check(len(state['draws']) == len(consumed)
              and all(n['utility_measures'] == own[c] for c, n in nodes.items())
              and state['admission'] == {n['commit_id']:i+1 for i,n in enumerate(state['nodes'])}
              and {sid for sid,s in state['slots'].items() if s['status'] == 'completed'} ==
                  {d['reservation_id'] for d in state['draws']}, 'FINAL_CHECKPOINT_OBSERVATIONS')
        if state['selector'] is not None:
            from wgm_v2_selector import LineageEXP3
            from final_adapter import config_from_manifest
            config = config_from_manifest(run.manifest)
            LineageEXP3.restore(state['selector'], config.wgm_v2)
            check(state['selector']['rng'] == state['v2_rng']['selection'], 'FINAL_CHECKPOINT_RNG')
    else:
        from hgm_incremental import check_evidence
        root_results = {r['task_id']: int(r['result'] == 'resolved') for r in cached['per_task_results']}
        direct = check_evidence(run, cp.row, root_results)
        from final_adapter import choose_legacy, selection_evidence
        selected = choose_legacy('hgm', list(nodes), direct)
        check(cp.row['final_selection'] == selection_evidence(cp, selected, identities[selected['candidate_id']]),
              'FINAL_CHECKPOINT_SELECTION')
    journal = SearchOperationJournal(cp.directory, run.manifest['run_id'], run.manifest['reproducibility_fingerprint'])
    operations = journal.plans()
    committed = context['consumed_operations']
    check({p.operation_id for p in operations[:len(committed)]} == set(committed)
          and len(committed) == len(state['history']), 'FINAL_CHECKPOINT_OPERATIONS')
    candidates = {}
    for plan, history in zip(operations[:len(committed)], state['history']):
        value, _ = journal.inspect(plan)
        terminal = value['terminal']
        check(terminal is not None and committed[plan.operation_id] == {
            'operation_terminal_digest': terminal['terminal_digest'], 'candidate_digest': terminal['candidate_digest']}
            and plan.identity['candidate_id'] == history['candidate_id']
            and plan.identity['budget']['expansion']['expansion_id'] == history['expansion_id'],
            'FINAL_CHECKPOINT_OPERATION')
        if terminal['outcome'] == 'candidate':
            artifact = journal.read_artifact(plan, terminal['artifact'])
            cid = plan.identity['candidate_id']
            chain = read_chain(cp.directory, cid, require_versioned=True)
            metadata = io._read_json(cp.directory / cid / 'metadata.json')
            check(cid in nodes and io.digest(chain[-1].provenance.to_dict()) == terminal['candidate_digest']
                  and {k:v for k,v in metadata.items() if k != 'overall_performance'} ==
                      {k:v for k,v in artifact['metadata'].items() if k != 'overall_performance'},
                  'FINAL_CHECKPOINT_PROVENANCE')
            candidates[cid] = terminal['candidate_digest']
    return {'ledger_digest': io.digest(heads), 'operations_digest': io.digest(committed),
            'candidate_digest': io.digest(candidates), 'rng_digest': io.digest(checkpoint_rng(state)),
            'tree_digest': io.digest(state['nodes']), 'selection_digest': io.digest(cp.selection)
                if hasattr(cp, 'selection') else cp.row['final_selection']['selection_digest']}
