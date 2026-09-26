"""V2 canonical SearchState: independent RNGs, IID observations and fixed lineages."""
from copy import deepcopy
from dataclasses import asdict
import random

import evaluation_ledger as io
from search_state import SearchState
from wgm_v2_config import identity, runtime_control
from wgm_v2_selector import LineageEXP3, bai, interval, restore_rng, rng_snapshot, seed_for


class WGMV2State(SearchState):
    def __init__(self, config, nodes, submitted, tasks):
        super().__init__(config, nodes, submitted, tasks)
        self.selection_rng = random.Random(seed_for('canonical-selection', config.search.seed))
        self.task_rng = random.Random(seed_for('canonical-task', config.wgm_v2.task_seed))
        self.founders = {0: 'initial'}
        self.admission = {'initial': 1}
        self.round = None
        self.reservation_counter = 0
        self.slots = {}
        self.draws = []
        self.stop_reason = None
        self.zero_progress_expansions = 0
        self.initialized_identity = None
        # Control history deliberately lives outside the canonical state digest.
        self.runtime_control = runtime_control(config)

    def identity(self):
        return identity(self.config)

    def rng_envelope(self):
        return {'schema_version': 2, 'algorithm': 'wgm_v2',
                'selection': rng_snapshot(self.selection_rng), 'task': rng_snapshot(self.task_rng)}

    def rows(self, lineage=None):
        rows = []
        for node in self.nodes.values():
            if lineage is not None and self.branches[node.id] != lineage:
                continue
            values = node.utility_measures
            if any(type(v) is not int or v not in (0, 1) for v in values):
                raise ValueError('V2_OBSERVATION_INVALID')
            rows.append({'candidate_id': node.commit_id, 'node_id': node.id,
                         'lineage_id': self.branches[node.id], 'admission_index': self.admission[node.commit_id],
                         **interval(self.admission[node.commit_id], len(values), sum(values), self.config.wgm_v2.delta)})
        return rows

    def choose_bai(self, lineage):
        return bai(self.rows(lineage), self.config.wgm_v2.epsilon)

    def reserve(self, run_id, expansion_id, domain, number):
        ids = []
        for _ in range(number):
            counter = self.reservation_counter
            sid = 'slot-' + io.digest({'run': run_id, 'algorithm': 'wgm_v2', 'domain': domain,
                                      'expansion': expansion_id, 'counter': counter})
            self.slots[sid] = {'reservation_index': counter, 'domain': domain,
                              'expansion_id': expansion_id, 'status': 'reserved'}
            self.reservation_counter += 1
            ids.append(sid)
        return ids

    def end_expansion(self, child):
        """Called by the existing N2 atomic admission transaction."""
        row, current = self.history[-1], self.round
        initial = row['phase'] == 'initialization'
        if child is None:
            row['status'] = 'failed'
            for sid in current['batch_slots']:
                self.slots[sid]['status'] = 'released_invalid'
            if not initial:
                self.selector.feedback(row['expansion_id'], 0)
                if len(self.execution['observations']) == current['start_count']:
                    self.zero_progress_expansions += 1
            self.round = None
        else:
            if child.commit_id != row['candidate_id'] or child.commit_id in self.admission:
                raise ValueError('V2_ADMISSION_IDENTITY')
            branch = len(self.founders) if initial else self.branches[child.parent_id]
            self.branches[child.id] = branch
            if initial:
                self.founders[branch] = child.commit_id
            self.admission[child.commit_id] = len(self.admission)+1
            row.update(status='admitted', child_id=child.id)
            current.update(phase='batch', child_id=child.id)
            for sid in current['batch_slots']:
                self.slots[sid]['candidate_id'] = child.commit_id
        self.initial_attempts += int(initial)
        self.inflight = None

    def freeze(self):
        self.selector = LineageEXP3(self.founders, self.config.wgm_v2, self.selection_rng)
        self.initialized = True
        self.initialized_identity = {'schema_version': 1, 'effective_C': len(self.founders),
            'eta': self.selector.eta, 'founders': {str(k): v for k, v in self.founders.items()},
            'intent_digest': io.digest(self.identity())}

    def observe_binding(self, result):
        plan, context = result.plan, self.execution
        eid = plan.identity.evaluation_id
        previous = context['consumed_bindings'].get(eid)
        if previous is not None:
            if previous != result.binding_digest:
                raise ValueError('V2_BINDING_CONFLICT')
            return False
        if result.outcome != 'success' or type(result.resolved) is not bool:
            raise ValueError('V2_REQUIRES_TRUSTED_OBSERVATION')
        pending = self.inflight
        if not pending or pending['evaluation_plan'] != asdict(plan):
            raise ValueError('V2_BINDING_INFLIGHT')
        if plan.identity.task_draw_index != len(self.draws)-1:
            raise ValueError('V2_BINDING_DRAW')
        node = self.nodes[pending['node_id']]
        if node.commit_id != plan.identity.candidate_id or pending['task_id'] != plan.identity.task_id:
            raise ValueError('V2_BINDING_CANDIDATE')
        sid = pending['slot_id']
        slot = self.slots[sid]
        if slot['evaluation_id'] != eid or slot['status'] != 'sampled':
            raise ValueError('V2_BINDING_SLOT')
        value = int(result.resolved)
        node.utility_measures.append(value)
        # Diagnostic compatibility view only. BAI always reads own evaluation observations.
        self.results[node.commit_id].setdefault(plan.identity.task_id, value)
        context['consumed_bindings'][eid] = result.binding_digest
        context['observations'][eid] = value
        slot.update(status='completed', observation=value, binding_digest=result.binding_digest)
        self.inflight = None
        if slot['domain'] == 'local_bai':
            self.round['local_completed'] += 1
        else:
            batch = [self.slots[x] for x in self.round['batch_slots']]
            if all(x['status'] == 'completed' for x in batch):
                if not self.round['initial']:
                    self.selector.feedback(self.round['expansion_id'], sum(x['observation'] for x in batch)/len(batch))
                self.round = None
        return True

    def summary(self):
        completed = [s for s in self.slots.values() if s['status'] == 'completed']
        return {'experiment': self.identity(), 'initial_attempts': self.initial_attempts,
            'initialization_complete': self.initialized, 'effective_C': len(self.founders) if self.initialized else None,
            'valid_search_evaluations': len(completed), 'task_draws': len(self.draws),
            'unique_tasks': len({d['task_id'] for d in self.draws}),
            'repeat_draws': len(self.draws)-len({d['task_id'] for d in self.draws}),
            'reserved': sum(s['status'] in ('reserved', 'sampled') for s in self.slots.values()),
            'completed_by_domain': {d: sum(s['domain'] == d for s in completed)
                                    for d in ('initialization', 'local_bai', 'child_batch')},
            'invalid_expansions': sum(r['status'] == 'failed' for r in self.history),
            'zero_progress_expansions': self.zero_progress_expansions,
            'stop_reason': self.stop_reason, 'D_design': None,
            'D_design_reason': 'design_feature_schema_not_frozen'}

    def state_dict(self, n_task_evals):
        return {'schema_version': 1, 'v2_schema': 1, 'experiment': self.identity(),
            'v2_rng': self.rng_envelope(), 'execution': deepcopy(self.execution),
            'nodes': [{'id': n.id, 'commit_id': n.commit_id, 'parent_id': n.parent_id,
                       'utility_measures': list(n.utility_measures)} for n in self.nodes.values()],
            'tasks': list(self.tasks), 'n_task_evals': n_task_evals,
            'initial_attempts': self.initial_attempts, 'initialized': self.initialized,
            'branches': {str(k): v for k, v in self.branches.items()},
            'founders': {str(k): v for k, v in self.founders.items()}, 'admission': dict(self.admission),
            'history': deepcopy(self.history), 'inflight': deepcopy(self.inflight),
            'selector': self.selector.state_dict() if self.selector else None,
            'round': deepcopy(self.round), 'reservation_counter': self.reservation_counter,
            'slots': deepcopy(self.slots), 'draws': deepcopy(self.draws),
            'initialized_identity': deepcopy(self.initialized_identity), 'stop_reason': self.stop_reason,
            'zero_progress_expansions': self.zero_progress_expansions,
            'summary': self.summary()}

    def restore(self, state, node_type):
        from search_operation_journal import validate_rng
        validate_rng(state['v2_rng'])
        if (state.get('v2_schema') != 1 or not io.same_json(state['experiment'], self.identity())
                or state['tasks'] != self.tasks):
            raise ValueError('V2_STATE_IDENTITY')
        selection, task = restore_rng(state['v2_rng']['selection']), restore_rng(state['v2_rng']['task'])
        selector = LineageEXP3.restore(state['selector'], self.config.wgm_v2) if state['selector'] else None
        if selector and rng_snapshot(selector.rng) != rng_snapshot(selection):
            raise ValueError('V2_SELECTOR_RNG_MISMATCH')
        rows = state['nodes']
        if [r['id'] for r in rows] != list(range(len(rows))) or len(state['admission']) != len(rows):
            raise ValueError('V2_NODE_IDENTITY')
        for index, row in enumerate(rows, 1):
            if state['admission'].get(row['commit_id']) != index or any(type(x) is not int or x not in (0, 1) for x in row['utility_measures']):
                raise ValueError('V2_ADMISSION_STATISTICS')
            if row['parent_id'] is not None and not 0 <= row['parent_id'] < row['id']:
                raise ValueError('V2_PARENT_IDENTITY')
        observations = state['execution']['observations']
        if (set(observations) != set(state['execution']['consumed_bindings'])
                or len(observations) != state['n_task_evals']
                or sum(len(r['utility_measures']) for r in rows) != state['n_task_evals']):
            raise ValueError('V2_BUDGET_MISMATCH')
        self.nodes.clear()
        for row in rows:
            node_type(**deepcopy(row))
        for node in self.nodes.values():
            if node.parent_id is not None:
                self.nodes[node.parent_id].add_child(node)
        self.selection_rng, self.task_rng, self.selector = selection, task, selector
        if self.selector:
            self.selector.rng = self.selection_rng
        for name in ('execution', 'initial_attempts', 'initialized', 'history', 'inflight', 'admission',
                     'round', 'reservation_counter', 'slots', 'draws', 'initialized_identity', 'stop_reason', 'zero_progress_expansions'):
            setattr(self, name, deepcopy(state[name]))
        self.branches = {int(k): v for k, v in state['branches'].items()}
        self.founders = {int(k): v for k, v in state['founders'].items()}
        self.results.clear()
        for slot in self.slots.values():
            if slot['status'] == 'completed':
                self.results[slot['candidate_id']].setdefault(slot['task_id'], slot['observation'])
        self.rows()  # Pure statistics validation; no legacy clade or global RNG access.
        return state['n_task_evals']

    def top_lineages(self):
        ranked = []
        for c in sorted(self.founders):
            rows = self.rows(c)
            if not rows:
                continue
            top = bai(rows, self.config.wgm_v2.epsilon)['leader']
            node, chain = self.nodes[top['node_id']], []
            while node is not None:
                chain.append(node.commit_id)
                node = self.nodes.get(node.parent_id)
            ranked.append({**top, 'parent_chain': list(reversed(chain)),
                           'selection_basis': 'maximum_search_lower_bound',
                           'tie_break': 'candidate_utf8_bytes', 'certificate': None})
        ranked.sort(key=lambda r: (-r['lower'], r['lineage_id']))
        return {'schema_version': 1, 'primary': ranked[0] if ranked else None,
                'secondary': ranked[1] if len(ranked) > 1 else None,
                'secondary_reason': None if len(ranked) > 1 else 'fewer_than_two_eligible_lineages',
                'lineage_tie_break': 'ascending_lineage_id', 'D_design': None,
                'D_design_reason': 'design_feature_schema_not_frozen',
                'benchmark_metrics': dict.fromkeys(('primary_solved_set', 'secondary_solved_set',
                    'exclusive_solves', 'union_coverage', 'intersection', 'Jaccard', 'outcome_vector_Hamming_distance')),
                'benchmark_status': 'not_executed', 'conditional_confidence': 'per_IID_stationary_trajectory'}
