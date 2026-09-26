"""Read-only common-root evidence for the external comparison adapter."""
from pathlib import Path

from evaluation_plan import check
from run_metadata import ROOT, file_hash, source_snapshot
from root_cache import agent_condition, same_condition, verify_root_cache, root_reference
from task_schedule import canonical_tasks, schedule_identity

DEFAULT_ROOT = ROOT / 'results/verified-root'
# The common root of each agent-provider condition (Codex when absent). A run
# never compares against another condition's root by default.
DEFAULT_ROOTS = {(None, False): DEFAULT_ROOT,
                 ('claude_code', False): ROOT / 'results/claude-root/verified-root',
                 ('claude_code', True): ROOT / 'results/claude-root-polyglot/verified-root'}


def default_root(manifest):
    from task_schedule import manifest_polyglot
    key = (manifest.get('codex', {}).get('provider'), manifest_polyglot(manifest))
    check(key in DEFAULT_ROOTS, 'COMPARISON_ROOT_PROVIDER')
    return DEFAULT_ROOTS[key]
FINAL_ORDER_HASH = 'bcc4567b75cb219a43e12fbd4a89cfe38c81976d11f826ac78b7a1447d032bc1'


def final_order_hash(polyglot=False):
    from task_schedule import benchmark
    return benchmark(polyglot)['order_hash_42']


def common_root(manifest, directory=None):
    """An explicit historical observation import, never a new validated binding."""
    directory = Path(directory or default_root(manifest)).resolve()
    cached = verify_root_cache(directory)
    shared = cached['compatibility']
    from task_schedule import manifest_polyglot
    polyglot = manifest_polyglot(manifest)
    tasks = canonical_tasks(ROOT, 42, polyglot=polyglot)
    schedule = schedule_identity(tasks, 42)
    check(schedule['task_order_hash'] == final_order_hash(polyglot), 'FINAL_CANONICAL_ORDER')
    check(shared['source'] == manifest['policy']['initial_source_snapshot']
          and shared['policy_sha256'] == manifest['policy']['downstream_instructions_sha256']
          and shared['model'] == manifest['codex']['configured_model']
          and shared['reasoning_effort'] == manifest['codex']['configured_reasoning_effort']
          and agent_condition(shared) == agent_condition(manifest['codex'])
          and shared['ordered_task_ids'] == tasks
          and shared['task_membership_hash'] == manifest['tasks']['task_membership_hash']
          and shared['task_order_hash'] == final_order_hash(polyglot)
          and shared['dataset'] == manifest['dataset'], 'COMPARISON_ROOT_IDENTITY')
    recorded = manifest.get('dgm_inputs', {}).get('root_cache_digest') or manifest.get(
        'root_initialization', {}).get('root_cache_digest')
    check(recorded is None or recorded == cached['content_digest'], 'COMPARISON_ROOT_CACHE_CHANGED')
    certificate = {**root_reference(cached), 'source_digest': shared['source']['sha256'],
        'policy_digest': shared['policy_sha256'], 'model': shared['model'],
        'reasoning': shared['reasoning_effort'], 'task_membership_hash': schedule['task_membership_hash'],
        'task_order_hash': final_order_hash(polyglot), 'seal_digest': cached['baseline_content_digest'],
        **agent_condition(shared)}
    return certificate, cached, directory


def hgm_root_input(config):
    directory = Path(config.paths.root_cache_dir).resolve()
    cached = verify_root_cache(directory)
    source = directory / 'initial/src'
    shared = cached['compatibility']
    check(config.search.algorithm == 'hgm' and config.search.hgm_profile == 'legacy_code'
          and shared['source'] == source_snapshot(source)
          and shared['policy_sha256'] == file_hash(source / 'prompts/downstream_instructions.txt')
          and shared['model'] == config.codex.model
          and shared['reasoning_effort'] == config.codex.reasoning_effort
          and same_condition(shared, config.codex)
          and shared['ordered_task_ids'] == canonical_tasks(ROOT, 42, polyglot=config.evaluation.polyglot),
          'HGM_COMMON_ROOT_IDENTITY')
    return source, cached
