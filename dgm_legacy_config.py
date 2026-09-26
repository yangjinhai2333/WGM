"""Isolated DGM identity; pause/runtime controls never seed the selector."""
from dataclasses import asdict, dataclass
from pathlib import Path
from search_config import SearchConfigurationError

CHECKPOINTS = (100, 200, 400, 800)
ROOT_SOURCE = 'results/verified-root/initial/src'


@dataclass(frozen=True)
class DGMConfig:
    planning_horizon: int = 800
    target_budget: int = 800
    checkpoints: tuple = CHECKPOINTS
    selfimprove_size: int = 2
    selfimprove_workers: int = 2
    choose_selfimproves_method: str = 'score_child_prop'
    update_archive: str = 'keep_all'
    test_more_threshold: float = .4
    full_eval_140: bool = False
    max_generations: int = 80  # Original outer-loop default, runtime stop control.
    shallow_eval: bool = False  # Existing upstream small-only smoke option.
    initial_agent_source: str = ROOT_SOURCE

    def __post_init__(self):
        if (type(self.planning_horizon) is not int or self.planning_horizon != 800
            or tuple(self.checkpoints) != CHECKPOINTS or self.selfimprove_size != 2
            or type(self.selfimprove_size) is not int or self.choose_selfimproves_method != 'score_child_prop'
            or self.update_archive != 'keep_all' or type(self.test_more_threshold) not in (float, int)
            or self.test_more_threshold != .4 or self.full_eval_140 is not False
            or type(self.shallow_eval) is not bool):
            raise SearchConfigurationError('DGM requires the frozen default legacy path with no_full_eval')
        for name in ('target_budget', 'selfimprove_workers', 'max_generations'):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise SearchConfigurationError('Invalid dgm.' + name)
        if self.target_budget > 800 or self.selfimprove_workers > 2:
            raise SearchConfigurationError('Invalid DGM budget/workers')


def identity(config):
    cfg = config.dgm
    params = asdict(cfg)
    for key in ('target_budget', 'max_generations', 'initial_agent_source'):
        params.pop(key)
    return {'algorithm': 'dgm', 'dgm_profile': 'legacy_code', 'evaluation_protocol': 'hgm_comparison_60',
        'seed': config.search.seed, 'parameters': params,
        'budget_unit': 'trusted_agent_task_benchmark_evaluation',
        'initial_root_observations': 'shared_verified_cache_excluded_from_search_budget',
        'final_selection_rule': 'empirical_mean_then_legacy_archive_order',
        'upstream_commit': 'a565fd2d1dca504ef5104a7cc0f3bdc4ab9b4fd2'}


def initial_source(config, root):
    from evaluation_plan import check
    source = Path(config.dgm.initial_agent_source)
    source = (Path(root)/source).resolve() if not source.is_absolute() else source.resolve()
    # Source and policy identity are verified against the sealed root cache (root_input).
    check((source/'prompts/downstream_instructions.txt').is_file(), 'DGM_INITIAL_POLICY_MISSING')
    return source
