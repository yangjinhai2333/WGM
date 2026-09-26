"""DGM legacy controller: frozen parents, N2 operations, staged fixed evaluations.

Archive appends in self-improvement completion order, as in upstream as_completed.
Complete child lifecycles overlap, sharing a run-global task evaluation pool.
No worker selects parents/entries or draws RNG. No hidden model retries.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import random
import shutil
import threading
from types import SimpleNamespace

import numpy as np
import evaluation_ledger as io
from evaluation_plan import EvaluationPlan, check, durable_json
from evaluation_scheduler import read_binding
from search_operation_journal import SearchOperationPlan, durable_record, validate_rng
from dgm_legacy_config import CHECKPOINTS
from dgm_legacy_selector import choose_selfimproves, is_compiled_self_improve, update_archive, best_belief


def rng_envelope(rng, seed):
    state = np.random.RandomState(seed).get_state()
    result = {'schema_version': 1, 'global': {'python': rng.getstate(),
        'numpy': [state[0], state[1].tolist(), int(state[2]), int(state[3]), float(state[4])]}, 'selector': None}
    return io._decode(io.canonical(result))


def restore_random(envelope):
    validate_rng(envelope)
    value = envelope['global']['python']
    rng = random.Random()
    rng.setstate((value[0], tuple(value[1]), value[2]))
    return rng


def read_checkpoint(directory, fingerprint=None):
    row = io._read_json(Path(directory)/'search_state.json')
    check(row['sha256'] == io.digest(row['state']) and row['state']['dgm_schema'] == 1, 'DGM_CHECKPOINT_DIGEST')
    check(fingerprint is None or row['fingerprint'] == fingerprint, 'DGM_CHECKPOINT_IDENTITY')
    validate_rng(row['state']['rng'])
    check(row['state']['valid_evaluations'] == len(row['state']['consumed_bindings']), 'DGM_BUDGET_IDENTITY')
    reserved = 0
    for generation in row['state']['generations']:
        validate_rng(generation['pre_rng'])
        validate_rng(generation['post_rng'])
        for child in generation['children']:
            SearchOperationPlan.from_dict(child['operation_plan'])
            for stage in child['stages'].values():
                check(type(stage.get('cursor')) is int and type(stage.get('scheduled')) is int
                      and 0 <= stage['cursor'] <= stage['scheduled'] <= len(stage['plans']), 'DGM_STAGE_BUDGET')
                reserved += stage['scheduled'] - stage['cursor']
    check(row['state']['valid_evaluations'] + reserved <= row['state']['target_budget'], 'DGM_STAGE_BUDGET')
    return row


class LockedSources:
    """Avoid two sibling preparations racing to reconstruct the same parent."""
    def __init__(self, sources):
        self.sources, self.root, self.mutex = sources, sources.root, threading.RLock()
    def get(self, candidate):
        with self.mutex:
            return self.sources.get(candidate)


class DGMSearch:
    def __init__(self, run, config, sources, entries, small, medium, operation_factory, execute,
                 *, fault=lambda point: None):
        self.run, self.config, self.sources, self.entries = run, config, sources, entries
        self.small, self.medium, self.operation_factory, self.execute = small, medium, operation_factory, execute
        self.mutex = threading.RLock()
        self.evaluation_pool = None
        self.fingerprint, self.fault = run.manifest['reproducibility_fingerprint'], fault
        self.state = None
        self.cancel = getattr(config.codex, 'readiness_cancel_event', None) or threading.Event()

    def read(self):
        return read_checkpoint(self.run.directory, self.fingerprint)

    def save(self):
        self.state['summary'] = {'algorithm': 'dgm', 'generation': self.state['generation'],
            'archive_size': len(self.state['archive']), 'valid_benchmark_evaluations': self.state['valid_evaluations']}
        payload = deepcopy(self.state)
        durable_json(self.run.directory/'search_state.json',
            {'state': payload, 'sha256': io.digest(payload), 'fingerprint': self.fingerprint},
            replace_existing=True, fault=self.fault)

    def change(self, action):
        with self.mutex:
            try:
                value = action()
                self.save()
                return value
            except BaseException:
                self.state = self.read()['state']
                raise

    def initialize(self, root_metadata):
        self.state = {'dgm_schema': 1, 'archive': ['initial'], 'node_ids': {'initial': 0},
            'metadata': {'initial': deepcopy(root_metadata)}, 'generation': 0, 'generations': [],
            'valid_evaluations': 0, 'consumed_bindings': {}, 'consumed_operations': {},
            'rng': rng_envelope(random.Random(self.config.search.seed), self.config.search.seed),
            'target_budget': self.config.dgm.target_budget, 'continuations': []}
        self.save()

    def resume(self):
        self.state = self.read()['state']
        old, new = self.state['target_budget'], self.config.dgm.target_budget
        check(new >= old, 'DGM_TARGET_DECREASE')
        if new != old:
            check(self.state['valid_evaluations'] == old, 'DGM_PREVIOUS_TARGET_INCOMPLETE')
            self.change(lambda: (self.state['continuations'].append({'from': old, 'to': new,
                'checkpoint_digest': self.read()['sha256']}), self.state.update(target_budget=new)))
        # Validate all consumed bindings and operation terminals before any call.
        frozen = [EvaluationPlan.from_dict(p) for g in self.state['generations'] for child in g['children']
                  for stage in child['stages'].values() for p in stage['plans']]
        for plan in sorted(frozen, key=lambda p: p.identity.ordinal):
            self.run.publish(plan)
        self.run.precheck(self.state['consumed_bindings'])
        plans = {p.identity.evaluation_id: p for p in self.run.plans.read()}
        for eid, digest in self.state['consumed_bindings'].items():
            check(eid in plans and read_binding(self.run.ledger, plans[eid]).binding_digest == digest,
                  'DGM_BINDING_CHANGED')
        for generation in self.state['generations']:
            for child in generation['children']:
                plan = SearchOperationPlan.from_dict(child['operation_plan'])
                op = self.operation_factory()
                op.journal.publish(plan)
                if child['operation_terminal'] is not None:
                    op.recovered(self, child)

    def start_generation(self):
        # The selector reads these exact metadata views, not a HGM Node/selector.
        for candidate in self.state['archive']:
            actual = io._read_json(self.run.directory/candidate/'metadata.json')
            check(actual == self.state['metadata'][candidate], 'DGM_PARENT_METADATA_CHANGED')
            self.sources.get(candidate)
        before = self.read()
        rng = restore_random(self.state['rng'])
        # Trace only the probabilities already supplied to choices; no extra draws.
        probabilities = []
        choices = rng.choices
        def traced(population, weights=None, *, cum_weights=None, k=1):
            probabilities.append({'candidates': list(population), 'probabilities': list(weights)})
            return choices(population, weights, cum_weights=cum_weights, k=k)
        rng.choices = traced
        selected = choose_selfimproves(str(self.run.directory), self.state['archive'], 2,
                                      method='score_child_prop', polyglot=self.config.evaluation.polyglot, rng=rng)
        post = rng_envelope(rng, self.config.search.seed)
        ordinal = sum(len(g['children']) for g in self.state['generations'])
        generation = {'index': self.state['generation'], 'archive_before': list(self.state['archive']),
            'parents_metadata': {p: deepcopy(self.state['metadata'][p]) for p, _ in selected},
            'pre_rng': deepcopy(self.state['rng']), 'post_rng': post, 'selection': probabilities[0],
            'selected_entries': [list(x) for x in selected], 'children': [], 'completion_order': [],
            'admitted': [], 'rejected': [], 'committed': False}
        op = self.operation_factory()
        for index, (parent, entry) in enumerate(selected):
            child = {'id': f'candidate-{ordinal+index:06d}', 'parent': parent, 'entry': entry,
                'operation_terminal': None, 'stages': {}, 'results': [], 'status': 'generation', 'admitted': False}
            child['operation_plan'] = op.make_plan(self, child, ordinal+index, generation['pre_rng'], post,
                                                   io.digest(before)).to_dict()
            generation['children'].append(child)
        # Publish both full plans and post-selection RNG before any worker or marker.
        self.change(lambda: (self.state['generations'].append(generation), self.state.update(rng=post)))
        for child in generation['children']:
            op.journal.publish(SearchOperationPlan.from_dict(child['operation_plan']))
        return generation

    def child(self, candidate):
        return next(c for g in self.state['generations'] for c in g['children'] if c['id'] == candidate)

    @property
    def reserved_evaluations(self):
        # Persist the granted prefix: after interruption, pending/bound work in
        # one sibling still owns its budget slots before the other can add work.
        return sum(s['scheduled'] - s['cursor'] for g in self.state['generations']
                   for c in g['children'] for s in c['stages'].values())

    def stage(self, candidate, name, tasks):
        from evaluation_batch import execute_batch
        # Only publication and budget reservation are serialized. Both child
        # stages then submit to the same bounded executor, without holding this lock.
        with self.mutex:
            child = self.child(candidate)
            if name not in child['stages']:
                source = self.sources.get(candidate)
                ordinal = len(self.run.plans.read())
                plans = [self.run.plan(self.entries[t], stage='search', candidate_id=candidate,
                    ordinal=ordinal+i, task_draw_index=ordinal+i,
                    selection_digest=io.digest({'operation_id': child['operation_plan']['operation_id'], 'stage': name}),
                    policy_fingerprint=source['policy_fingerprint'], source_fingerprint=source['source_fingerprint'])
                    for i,t in enumerate(tasks)]
                self.change(lambda: self.child(candidate)['stages'].update({name: {
                    'plans': [asdict(p) for p in plans], 'cursor': 0, 'scheduled': 0}}))
            current = self.child(candidate)['stages'][name]
            plans = [EvaluationPlan.from_dict(p) for p in current['plans']]
            for plan in plans:
                self.run.publish(plan)
            start = current['cursor']
            remaining = self.state['target_budget'] - self.state['valid_evaluations'] - self.reserved_evaluations
            check(remaining >= 0 and start <= current['scheduled'] <= len(plans), 'DGM_STAGE_BUDGET')
            stop = min(len(plans), current['scheduled'] + remaining)
            if stop != current['scheduled']:
                self.change(lambda: self.child(candidate)['stages'][name].update(scheduled=stop))
            chunk = plans[start:stop]
        def consume(plan, result):
            check(result.outcome == 'success' and type(result.resolved) is bool, 'DGM_TRUSTED_RESULT_REQUIRED')
            def record():
                eid = plan.identity.evaluation_id
                if eid in self.state['consumed_bindings']:
                    check(self.state['consumed_bindings'][eid] == result.binding_digest, 'DGM_BINDING_CHANGED')
                    return
                child = self.child(candidate)
                check(child['stages'][name]['cursor'] == plans.index(plan), 'DGM_TASK_CURSOR')
                child['results'].append({'evaluation_id': eid, 'task_id': plan.identity.task_id,
                    'stage': name, 'resolved': result.resolved, 'empty_patch': result.empty_patch})
                child['stages'][name]['cursor'] += 1
                self.state['consumed_bindings'][eid] = result.binding_digest
                self.state['valid_evaluations'] += 1
            with self.mutex:
                self.change(record)
                self.milestone()
        if chunk:
            execute_batch(self.run, chunk, self.execute, workers=self.config.execution.evaluation_workers,
                          consume=consume, cancel_event=self.cancel, shared_pool=self.evaluation_pool)
        return self.child(candidate)['stages'][name]['cursor'] == len(tasks)

    def attempt(self, candidate):
        if self.cancel.is_set():
            raise KeyboardInterrupt
        child = self.child(candidate)
        op = self.operation_factory()  # Each attempt owns its runner/readiness authority.
        terminal = op.recovered(self, child)
        if child['operation_terminal'] is None:
            def committed():
                self.child(candidate)['operation_terminal'] = terminal
                self.state['consumed_operations'][child['operation_plan']['operation_id']] = {
                    'operation_terminal_digest': terminal['terminal_digest'], 'candidate_digest': terminal['candidate_digest']}
            self.change(committed)
        if terminal['outcome'] != 'candidate':
            self.change(lambda: self.child(candidate).update(status='rejected'))
            return candidate
        if not self.stage(candidate, 'small', self.small):
            return None
        small_resolved = sum(r['resolved'] for r in self.child(candidate)['results'] if r['stage'] == 'small')
        if not self.config.dgm.shallow_eval and small_resolved >= len(self.small)*.4:
            if not self.stage(candidate, 'medium', self.medium):
                return None
        self.change(lambda: self.child(candidate).update(status='evaluated'))
        return candidate

    def finalize_generation(self):
        generation = self.state['generations'][-1]
        if generation['committed']:
            return True
        if len(generation['completion_order']) != len(generation['children']):
            return False
        from validated_runs import refresh_views
        refresh_views(self.run, candidates={c['id'] for c in generation['children']})
        metadata = {}
        admitted, rejected = [], []
        logger = SimpleNamespace(info=lambda *args: None)
        for cid in generation['completion_order']:
            child = self.child(cid)
            if child['status'] != 'evaluated':
                rejected.append(cid)
                continue
            row = io._read_json(self.run.directory/cid/'metadata.json')
            results = child['results']
            resolved = [r['task_id'] for r in results if r['resolved']]
            empty = [r['task_id'] for r in results if r['empty_patch'] and not r['resolved']]
            unresolved = [r['task_id'] for r in results if not r['resolved'] and not r['empty_patch']]
            row['overall_performance'].update(accuracy_score=len(resolved)/len(results),
                total_resolved_instances=len(resolved), total_submitted_instances=len(results),
                total_resolved_ids=resolved, total_unresolved_ids=unresolved, total_emptypatch_ids=empty)
            durable_json(self.run.directory/cid/'metadata.json', row, replace_existing=True)
            metadata[cid] = row
            (admitted if is_compiled_self_improve(row, [10,50], logger) else rejected).append(cid)
        def finish():
            generation = self.state['generations'][-1]
            generation.update(admitted=admitted, rejected=rejected, committed=True)
            self.state['archive'] = update_archive(str(self.run.directory), self.state['archive'], admitted)
            self.state['metadata'].update(metadata)
            for cid in admitted:
                self.state['node_ids'][cid] = len(self.state['node_ids'])
                self.child(cid)['admitted'] = True
            self.state['generation'] += 1
        self.change(finish)
        return True

    def milestone(self):
        with self.mutex:
            if self.state['valid_evaluations'] in CHECKPOINTS:
                path = self.run.directory/'checkpoints'/f"{self.state['valid_evaluations']:06d}.json"
                path.parent.mkdir(exist_ok=True)
                if not path.exists():
                    durable_record(path, self.read())
                frozen = io._read_json(path)
                best = best_belief(frozen['state']['archive'], frozen['state']['metadata'])
                selection = {'checkpoint_digest': frozen['sha256'],
                    'final_selection_rule': 'empirical_mean_then_legacy_archive_order',
                    'candidate_id': best, 'archive': frozen['state']['archive'],
                    'performance': frozen['state']['metadata'][best]['overall_performance']}
                selected = path.with_suffix('.selection.json')
                if selected.exists():
                    check(io._read_json(selected) == selection, 'DGM_CHECKPOINT_SELECTION_CHANGED')
                else:
                    durable_record(selected, selection)

    def run_to_budget(self):
        from evaluation_batch import _interrupts, SharedEvaluationPool
        with _interrupts(self.cancel), SharedEvaluationPool(self.run,
                self.config.execution.evaluation_workers, self.cancel, self.mutex) as self.evaluation_pool:
            while self.state['generation'] < self.config.dgm.max_generations:
                if self.cancel.is_set():
                    raise KeyboardInterrupt
                current = self.state['generations'][-1] if self.state['generations'] else None
                if current is None or current['committed']:
                    if self.state['valid_evaluations'] >= self.state['target_budget']:
                        break
                    current = self.start_generation()
                # Complete/reconcile the frozen generation even at a task budget boundary.
                todo = [c['id'] for c in current['children'] if c['id'] not in current['completion_order']]
                error = None
                with ThreadPoolExecutor(max_workers=self.config.dgm.selfimprove_workers) as pool:
                    futures = [pool.submit(self.attempt, candidate) for candidate in todo]
                    for future in as_completed(futures):
                        try:
                            cid = future.result()
                            if cid is not None:
                                self.change(lambda: self.state['generations'][-1]['completion_order'].append(cid))
                        except BaseException as exc:
                            from cancellation import request, primary
                            request(self.cancel, exc)
                            error = primary(self.cancel, error)
                if error:
                    raise error
                complete = self.finalize_generation()
                if not complete or self.state['valid_evaluations'] >= self.state['target_budget']:
                    break
        self.milestone()


def task_lists(root, polyglot=False):
    from task_schedule import benchmark, task_hash
    folder = Path(root)/('polyglot/subsets' if polyglot else 'swe_bench/subsets')
    small = json.loads((folder/'small.json').read_text())
    medium = json.loads((folder/'medium.json').read_text())
    if polyglot:
        # Polyglot medium50 repeats one small10 task; each task is evaluated once (small first).
        medium = [t for t in medium if t not in small]
    frozen = benchmark(polyglot)
    check(len(small) == 10 and len(set(small+medium)) == len(small+medium) == frozen['count']
          and task_hash(sorted(small+medium)) == frozen['membership_hash'], 'DGM_TASK_MEMBERSHIP')
    return small, medium


# The frozen Codex condition of the original comparison. Other providers read
# their condition from the run's configuration, which must be pinned and equal
# the root cache's (provider, model, reasoning effort, prompt identity).


def root_condition_matches(shared, config):
    from root_cache import same_condition
    if getattr(config.codex, 'provider', None) is None:
        # Codex condition: the pinned model and effort must equal the verified root's.
        return (bool(config.codex.model) and config.codex.reasoning_effort is not None and 'provider' not in shared
                and (shared['model'], shared['reasoning_effort']) == (config.codex.model, config.codex.reasoning_effort))
    return (bool(config.codex.model) and config.codex.reasoning_effort is not None
            and shared['model'] == config.codex.model and shared['reasoning_effort'] == config.codex.reasoning_effort
            and same_condition(shared, config.codex))


def root_input(config, root):
    from dgm_legacy_config import initial_source
    from root_cache import verify_root_cache
    from run_metadata import file_hash, source_snapshot
    source = initial_source(config, root)
    cache = source.parent.parent
    cached = verify_root_cache(cache)
    shared = cached['compatibility']
    small, medium = task_lists(root, config.evaluation.polyglot)
    check(shared['source'] == source_snapshot(source)
          and shared['policy_sha256'] == file_hash(source/'prompts/downstream_instructions.txt')
          and set(shared['ordered_task_ids']) == set(small+medium)
          and root_condition_matches(shared, config), 'DGM_ROOT_CACHE_IDENTITY')
    return source, cached


def manifest_inputs(manifest, config, entries, cached, root):
    from run_metadata import file_hash, fingerprint_inputs, digest
    from task_schedule import task_hash, canonical_tasks
    small, medium = task_lists(root, config.evaluation.polyglot)
    manifest['dgm_inputs'] = {'schema_version': 1, 'root_cache_digest': cached['content_digest'],
        'root_cache_creator_commit': cached['compatibility']['creator_commit'],
        'root_observation_provenance': 'explicit_shared_historical_verified_root_not_new_validated_bindings',
        'root_metadata_sha256': file_hash(Path(config.dgm.initial_agent_source).resolve().parent/'metadata.json'),
        'root_resolved': cached['resolved'], 'root_evaluations': len(small+medium), 'root_budget_charge': 0,
        'small': small, 'medium': medium, 'dgm_order_hash': task_hash(small+medium),
        'common_membership_hash': task_hash(sorted(small+medium)),
        'hgm_order_hash': task_hash(canonical_tasks(root, 42, polyglot=config.evaluation.polyglot)),
        'entry_hashes': {t: io.digest(entries[t]) for t in small+medium},
        'implementation_hashes': {p: file_hash(Path(root)/p) for p in ('dgm_legacy_config.py',
            'dgm_legacy_selector.py', 'dgm_legacy_search.py', 'dgm_legacy_operations.py', 'dgm_legacy_prompts.py')},
        'upstream_files': {'DGM_outer.py': '239bc76f0e1b78210b8f7b3757b1082f6ca07db3537d6af50a45bf97709795ed',
            'self_improve_step.py': '6f2ec364b11e7f6d35391ea66896fa2394cbca3bc59d68cf7352c2a77aa8cca7',
            'utils/evo_utils.py': '0eb779270390f39aa0c4ff045ec9b6c3049736c6a008ae3d2e7a2df7ef253570',
            'prompts/self_improvement_prompt.py': '7d061402038e9ccbb120088ac849863e5880a657ad4a60122d93fdd180e908f2'}}
    manifest['dgm_runtime'] = {'target_budget': config.dgm.target_budget,
        'max_generations': config.dgm.max_generations, 'protocol': config.protocol.effective_name,
        'formal_comparison_eligible': config.protocol.effective_name == 'formal' and not config.dgm.shallow_eval}
    manifest['fingerprint_inputs'] = fingerprint_inputs(manifest)
    manifest['reproducibility_fingerprint'] = digest(manifest['fingerprint_inputs'])
    return manifest


def run_search(config, *, preflight):
    from dgm_legacy_operations import DGMOperations
    from execution_policy import require_supported
    from run_metadata import ROOT, build_manifest, RunRecorder
    from validated_runs import Sources, _entry_map, _open, executor
    config.validate_search()
    require_supported(config)
    check(config.search.algorithm == 'dgm' and config.search.hgm_profile == 'legacy_code', 'DGM_PROFILE_REQUIRED')
    check(config.llm.diagnose_backend == config.llm.self_improve_backend == 'codex_exec', 'DGM_CODEX_REQUIRED')
    check(not config.candidate.validation_command, 'DGM_CUSTOM_ADMISSION_FORBIDDEN')
    check(config.protocol.effective_name == 'smoke' or config.dgm.target_budget in CHECKPOINTS, 'DGM_FORMAL_TARGET')
    check(not config.dgm.shallow_eval or config.protocol.effective_name == 'smoke', 'DGM_SHALLOW_SMOKE_ONLY')
    config.execution.max_task_evals = 800
    directory = config.paths.continue_from or config.paths.output_dir
    check(bool(directory), 'DGM_EXPLICIT_OUTPUT_REQUIRED')
    check(not config.paths.continue_from or not config.paths.output_dir or
          Path(directory).absolute() == Path(config.paths.output_dir).absolute(), 'DGM_RESUME_DIRECTORY')
    source, cached = root_input(config, ROOT)
    small, medium = task_lists(ROOT, config.evaluation.polyglot)
    entries = _entry_map(config, small+medium)
    proposed = manifest_inputs(build_manifest(config, directory, 'dgm', small+medium, source=source,
        require_clean=True, require_pinned_model=True, shuffle=False, seed=42), config, entries, cached, ROOT)
    resume = bool(config.paths.continue_from)
    if resume:
        saved = io._read_json(Path(directory)/'run_manifest.json')
        check(saved['reproducibility_fingerprint'] == proposed['reproducibility_fingerprint'], 'DGM_RESUME_IDENTITY')
        check(saved['dgm_runtime']['protocol'] == config.protocol.effective_name, 'DGM_PROTOCOL_CHANGED')
        row = read_checkpoint(directory, saved['reproducibility_fingerprint'])
        old, new = row['state']['target_budget'], config.dgm.target_budget
        check(new == old or new > old and row['state']['valid_evaluations'] == old, 'DGM_CONTINUATION_TARGET')
    else:
        check(not Path(directory).exists(), 'RUN_DIRECTORY_EXISTS')
    preflight(config, output_root=directory)
    run = _open(config, directory, proposed, resume)
    with run.lock():
        sources = LockedSources(Sources(run, source))
        if not resume:
            # Import the explicitly requested common initial evidence, never rewrite its origin.
            for name in ('metadata.json', 'predictions'):
                origin, destination = source.parent/name, run.directory/'initial'/name
                if origin.is_dir():
                    shutil.copytree(origin, destination)
                else:
                    shutil.copyfile(origin, destination)
        recorder = RunRecorder(run.directory, run.manifest, network_readiness=config.network_readiness)
        cancel = getattr(config.codex, 'readiness_cancel_event', None) or threading.Event()
        config.codex.readiness_cancel_event = cancel
        def operations():
            return DGMOperations(run, config, sources, entries, recorder)
        engine = DGMSearch(run, config, sources, entries, small, medium, operations,
                          executor(config, run, entries, sources, recorder, cancel_event=cancel))
        if resume:
            engine.resume()
        else:
            engine.initialize(io._read_json(run.directory/'initial/metadata.json'))
        run.manifest['dgm_runtime'].update(target_budget=config.dgm.target_budget,
            max_generations=config.dgm.max_generations, continuations=engine.state['continuations'])
        durable_json(run.directory/'run_manifest.json', run.manifest, replace_existing=True)
        error = None
        try:
            engine.run_to_budget()
        except BaseException as exc:
            error = getattr(exc, 'code', 'DGM_INTERRUPTED')
            raise
        finally:
            complete = not error and engine.state['valid_evaluations'] == engine.state['target_budget']
            recorder.refresh('completed' if complete else 'incomplete')
            summary = report(run.directory, run.manifest)
            summary['error_code'] = error
            durable_json(run.directory/'dgm_summary.json', summary, replace_existing=True)
            durable_json(run.directory/'best_belief.json', {
                'checkpoint_digest': summary['checkpoint_digest'],
                'final_selection_rule': summary['final_selection_rule'], **summary['best_belief']}, replace_existing=True)
    return run.directory


def report(directory, manifest):
    from evaluation_scheduler import ExecutionRun
    from search_operation_journal import SearchOperationJournal
    from wgm_v2_search import physical_accounting
    from wgm_v2_derived import accounting
    from evaluation_batch import batch_reports
    directory = Path(directory)
    run = ExecutionRun(directory, manifest)
    row = read_checkpoint(directory, manifest['reproducibility_fingerprint'])
    state = row['state']
    from candidate_provenance import read_chain
    for candidate in state['archive']:
        check(io._read_json(directory/candidate/'metadata.json') == state['metadata'][candidate],
              'DGM_REPORT_ARCHIVE_CHANGED')
        read_chain(directory, candidate, require_versioned=True)
    plans = {p.identity.evaluation_id: p for p in run.plans.read()}
    for eid, digest in state['consumed_bindings'].items():
        result = read_binding(run.ledger, plans[eid])
        check(result.binding_digest == digest and result.outcome == 'success', 'DGM_REPORT_BINDING')
    journal = SearchOperationJournal(directory, manifest['run_id'], manifest['reproducibility_fingerprint'])
    for plan in journal.plans():
        operation, _ = journal.inspect(plan)
        if plan.operation_id in state['consumed_operations']:
            terminal = operation['terminal']
            check(terminal and state['consumed_operations'][plan.operation_id] == {
                'operation_terminal_digest': terminal['terminal_digest'], 'candidate_digest': terminal['candidate_digest']},
                'DGM_REPORT_OPERATION')
    calls = physical_accounting(run, SimpleNamespace(journal=journal))
    calls['operator_waived_unknown_model_attempts'] = sum(
        e.kind == 'attempt_abandoned' for snapshot in run.ledger.evaluations() for e in snapshot.events)
    timing = accounting(directory)
    saved = io._read_json(directory/'run_summary.json') if (directory/'run_summary.json').exists() else {}
    best = best_belief(state['archive'], state['metadata'])
    generations = []
    for g in state['generations']:
        generations.append({'generation': g['index'], 'selected_parents': [x[0] for x in g['selected_entries']],
            'entries': [x[1] for x in g['selected_entries']], 'parent_probabilities': g['selection'],
            'completion_order': g['completion_order'], 'admitted': g['admitted'], 'rejected': g['rejected'],
            'children': [{'candidate_id': c['id'], 'status': c['status'], 'admitted': c['admitted'],
                **{stage: {'submitted': len(c['stages'].get(stage, {}).get('plans', [])),
                    'completed': sum(r['stage'] == stage for r in c['results']),
                    'resolved': sum(r['resolved'] for r in c['results'] if r['stage'] == stage)}
                   for stage in ('small', 'medium')}} for c in g['children']]})
    network = directory/'network_readiness.json'
    return {'report_type': 'dgm_legacy_search', 'algorithm': 'dgm', 'dgm_profile': 'legacy_code',
        'evaluation_protocol': 'hgm_comparison_60', 'status': saved.get('status', 'incomplete'),
        'commit': manifest['repository']['commit'], 'execution_policy': manifest['execution_policy']['effective_policy'],
        'model': manifest['codex']['configured_model'], 'reasoning': manifest['codex']['configured_reasoning_effort'],
        'effective_config': manifest['experiment'], 'runtime': manifest.get('dgm_runtime'),
        'generation': state['generation'], 'archive_size': len(state['archive']), 'archive': state['archive'],
        'archive_digest': io.digest([{'id': c, 'metadata': state['metadata'][c]} for c in state['archive']]),
        'checkpoint_digest': row['sha256'], 'rng_state_digest': io.digest(state['rng']),
        'planning_horizon': 800, 'target_budget': state['target_budget'],
        'target_completed': state['valid_evaluations'] == state['target_budget'],
        'stop_reason': ('target_budget' if state['valid_evaluations'] == state['target_budget'] else
            'generation_limit' if state['generation'] >= manifest.get('dgm_runtime', {}).get('max_generations', 80)
            else 'incomplete'),
        'valid_benchmark_evaluations': state['valid_evaluations'], 'initial_cached_evaluations': 60,
        'calls': calls, 'generations': generations,
        'timing': {**timing, 'search_wall_seconds': saved.get('timing', {}).get('whole_run_total')},
        'evaluation_parallelism': manifest.get('evaluation_parallelism'), 'evaluation_batches': batch_reports(directory),
        'selfimprove_workers': manifest['experiment']['parameters']['selfimprove_workers'],
        'network_readiness': io._read_json(network) if network.exists() else None,
        'final_selection_rule': 'empirical_mean_then_legacy_archive_order',
        'best_belief': {'candidate_id': best, 'performance': state['metadata'][best]['overall_performance']},
        'final60': 'not_executed', 'production_ready': False}


def report_text(summary):
    return json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
