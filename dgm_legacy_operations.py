"""Thin DGM generation-plan adapter to the existing N1/N2 lifecycle."""
from copy import deepcopy
import hashlib

import evaluation_ledger as io
from search_operations import SearchOperations
from search_operation_journal import SearchOperationPlan, require
from search_operation_workspace import verify_source_links


class DGMOperations(SearchOperations):
    def publish_plan(self, plan, engine):
        with engine.mutex:
            self.journal.publish(plan)

    def prompt(self, entry, parent):
        from search_operations import diagnosis_prompt
        from dgm_legacy_prompts import get_diagnose_prompt_swe
        if self.prompt_builder is not diagnosis_prompt:
            return super().prompt(entry, parent)
        return diagnosis_prompt(self.config, self.sources, self.run.directory, self.entries, entry, parent,
                                legacy_builder=get_diagnose_prompt_swe)

    def make_plan(self, engine, child, ordinal, pre, post, checkpoint_digest):
        parent = child['parent']
        source = self.sources.get(parent)
        entry = child['entry']
        task, row = self.task(entry, parent)
        fingerprints = self.fingerprints(source)
        prompt = self.prompt(entry, parent)
        expansion = {'expansion_id': f'expansion-{ordinal:06d}', 'candidate_id': child['id'],
            'parent_id': engine.state['node_ids'][parent], 'selected_branch_id': 0,
            'selection_probability': None, 'child_id': None, 'phase': 'search', 'target_M': 10, 'status': 'creating'}
        return SearchOperationPlan.create(run_id=self.journal.run_id, ordinal=ordinal, algorithm='dgm',
            profile='legacy_code', policy='validated_observation_v2', run_fingerprint=self.journal.fingerprint,
            parent={'node_id': engine.state['node_ids'][parent], 'candidate_id': parent, 'branch_id': 0,
                'provenance_digest': source['chain_digest'], 'source_digest': source['source_fingerprint']},
            candidate_id=child['id'],
            diagnosis={'entry': entry, 'task_id': task, 'entry_digest': io.digest(row), 'commit': parent,
                'base_commit': row.get('base_commit'), 'prompt_digest': hashlib.sha256(prompt.encode()).hexdigest(),
                'max_attempts': 4},
            generation={'prompt_recipe_digest': io.digest({'builder': fingerprints['builder'], 'schema': 1,
                'test_description': 'swe_self_improvement', 'candidate_id': child['id']})},
            fingerprints=fingerprints, budget={'used': engine.state['valid_evaluations'], 'limit': 800,
                'initial_attempts': 0, 'initialized': True, 'expansion': expansion},
            pre_rng=pre, post_rng=post, pre_rng_digest=io.digest(pre), post_rng_digest=io.digest(post),
            checkpoint_digest=checkpoint_digest)

    def verify(self, plan, engine, session=None, *, committed=False):
        # A sibling may finish while this operation runs. The frozen generation,
        # parent and RNG must match; live sibling budget counters need not match.
        with engine.mutex:
            require(committed or not engine.cancel.is_set(), 'DGM_CANCELLED')
            self.journal.validate(plan)
            self.verify_runners()
            saved = engine.read()
            require(saved['state'] == engine.state, 'BLOCK_IDENTITY_MISMATCH')
            found = [(g, c) for g in engine.state['generations'] for c in g['children']
                     if c['id'] == plan.identity['candidate_id']]
            require(len(found) == 1, 'BLOCK_IDENTITY_MISMATCH')
            generation, child = found[0]
            require(child['operation_plan'] == plan.to_dict()
                    and generation['post_rng'] == plan.identity['post_rng']
                    and generation['pre_rng'] == plan.identity['pre_rng'], 'BLOCK_IDENTITY_MISMATCH')
            require(child['parent'] in generation['archive_before']
                    and engine.state['metadata'][child['parent']] == generation['parents_metadata'][child['parent']],
                    'BLOCK_IDENTITY_MISMATCH')
            require(io._read_json(self.run.directory/child['parent']/'metadata.json') ==
                    generation['parents_metadata'][child['parent']], 'BLOCK_IDENTITY_MISMATCH')
            source = self.sources.get(child['parent'])
            verify_source_links(source['path'])
            require(self.fingerprints(source) == plan.identity['fingerprints']
                    and source['chain_digest'] == plan.identity['parent']['provenance_digest'], 'BLOCK_IDENTITY_MISMATCH')
            row = self.entries.get(plan.identity['diagnosis']['task_id'], {})
            require(io.digest(row) == plan.identity['diagnosis']['entry_digest'], 'BLOCK_IDENTITY_MISMATCH')
            if session is not None:
                session.refresh()

    def recovered(self, engine, child):
        plan = SearchOperationPlan.from_dict(child['operation_plan'])
        if child['operation_terminal'] is None:
            return self.execute(plan, engine)
        with self.journal.session(plan) as session:
            self.verify(plan, engine, session, committed=True)
            require(child['operation_terminal'] == session.state['terminal'], 'BLOCK_IDENTITY_MISMATCH')
            self.workspace.publish(session, session.state['terminal'], committed=True)
            return deepcopy(session.state['terminal'])
