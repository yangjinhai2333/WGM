"""Frozen WGM V2 parameters. Pause targets are runtime controls, not horizons."""
from dataclasses import asdict, dataclass
import math
import platform
from pathlib import Path
import re

from search_config import SearchConfigurationError

SPEC = 'wgm-exp3-bai-v2-spec/1'
PROFILE = 'wgm_exp3_bai_v2'
LEGACY_CHECKPOINTS = (20, 100, 200, 800)
CHECKPOINTS = (20, 100, 200, 400, 800)
RNG_SCHEMA = 'python-random-mt19937-state-v3/v1'


@dataclass(frozen=True)
class WGMV2Config:
    initial_agent_source: str | None = None
    initial_source_sha256: str | None = None
    initial_policy_sha256: str | None = None
    planning_horizon: int = 800
    target_budget: int = 800
    smoke_target_budget: int = 8
    checkpoints: tuple = CHECKPOINTS
    initial_expansion_target: int = 5
    gamma: float = 0.1
    eta: float | None = None
    M: int = 1
    rho_r: float = 0.1
    delta: float = 0.05
    epsilon: float = 0.05
    local_bai_evals: int = 1
    checkpoint_bai_fraction: float = 0.2
    task_seed: int = 42
    safety_attempt_cap: int = 800

    def __post_init__(self):
        if self.initial_agent_source is not None and (not isinstance(self.initial_agent_source, str)
                or not self.initial_agent_source.strip() or not self.initial_source_sha256 or not self.initial_policy_sha256):
            raise SearchConfigurationError('V2_INITIAL_SOURCE_CONFIG_INVALID: explicit source needs source and policy digests')
        for value in (self.initial_source_sha256, self.initial_policy_sha256):
            if value is not None and (not isinstance(value, str) or re.fullmatch(r'[0-9a-f]{64}', value) is None):
                raise SearchConfigurationError('V2_INITIAL_SOURCE_DIGEST_INVALID')
        if type(self.planning_horizon) is not int or self.planning_horizon != 800:
            raise SearchConfigurationError('V2 planning_horizon is fixed at 800')
        if type(self.target_budget) is not int or self.target_budget not in CHECKPOINTS:
            raise SearchConfigurationError('V2 target_budget must be 20, 100, 200, 400 or 800')
        # Historical manifests retain their exact scientific identity. Only the
        # scoped continuation adapter authorizes a new boundary for an old run.
        if not isinstance(self.checkpoints, (list, tuple)) or any(type(n) is not int for n in self.checkpoints) or tuple(self.checkpoints) not in (CHECKPOINTS, LEGACY_CHECKPOINTS):
            raise SearchConfigurationError('V2 checkpoints must match a supported exact schedule')
        object.__setattr__(self, 'checkpoints', tuple(self.checkpoints))
        for name in ('initial_expansion_target', 'M', 'safety_attempt_cap', 'smoke_target_budget', 'local_bai_evals', 'task_seed'):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name in ('local_bai_evals', 'task_seed') else 1):
                raise SearchConfigurationError('Invalid V2 integer: ' + name)
        if self.smoke_target_budget > 800 or self.task_seed >= 2**32 or self.M > 800 or self.initial_expansion_target * self.M > 800:
            raise SearchConfigurationError('V2 initialization/seed exceeds bounds')
        for name in ('gamma', 'eta', 'rho_r', 'delta', 'epsilon', 'checkpoint_bai_fraction'):
            value = getattr(self, name)
            if name == 'eta' and value is None:
                continue
            if type(value) not in (float, int) or not math.isfinite(value):
                raise SearchConfigurationError('Invalid V2 number: ' + name)
            if name == 'eta':
                good = value > 0
            elif name == 'delta':
                good = 0 < value < 1
            else:
                good = (0 < value <= 1) if name == 'gamma' else (0 <= value <= 1)
            if not good:
                raise SearchConfigurationError('Invalid V2 range: ' + name)

    def scientific(self):
        value = asdict(self)
        for name in ('target_budget', 'smoke_target_budget', 'safety_attempt_cap', 'initial_agent_source'):
            value.pop(name)
        value['checkpoints'] = list(self.checkpoints)
        return value


def identity(config):
    return {'algorithm': 'wgm_v2', 'profile': PROFILE, 'spec_version': SPEC,
            'implementation_schema': 2, 'budget_semantics': 'fixed_horizon_increasing_pause_v1',
            'parameters': config.wgm_v2.scientific(), 'seed': config.search.seed,
            'rng_schema': RNG_SCHEMA, 'python_version': platform.python_version(),
            'interval': 'anytime_hoeffding_admission_index_v1',
            'tie_break': 'candidate_utf8_bytes_then_lineage_integer',
            'sampling': 'uniform_with_replacement', 'admission': 'corrected_structure_v1',
            'reward': 'immediate_child_batch_or_trusted_invalid_zero',
            'protocol': {'name': 'wgm_v2_canonical'},
            'requested_execution_policy': getattr(config, '_v2_requested_policy', config.execution.policy),
            'workers': 1, 'checkpoint_bai_execution': False, 'final_benchmark_execution': False,
            'D_design': None}


def pause_target(config):
    return (config.wgm_v2.smoke_target_budget if config.protocol.effective_name == 'smoke'
            else config.wgm_v2.target_budget)


def runtime_control(config):
    return {'schema_version': 1, 'protocol': config.protocol.effective_name,
            'target_budget': pause_target(config), 'continuations': [],
            'zero_progress_expansion_cap': config.wgm_v2.safety_attempt_cap}


def resolve_initial_source(config, root):
    """Select an existing frozen source. Never build an agent or fill files from the repository."""
    name = config.paths.initial_agent_name
    if not isinstance(name, str) or not name or Path(name).name != name or name in ('.', '..'):
        raise SearchConfigurationError('V2_INITIAL_AGENT_NAME_REQUIRED: set paths.initial_agent_name')
    configured = config.wgm_v2.initial_agent_source
    source = Path(root) / configured if configured else Path(root) / 'initial_swe' / name / 'src'
    validate_initial_source(config, source)
    return source


def validate_initial_source(config, source):
    """Fail before environment probes, data loading, model preparation or manifest publication."""
    from run_metadata import source_snapshot, file_hash
    source = Path(source)
    if not source.is_dir():
        raise SearchConfigurationError('V2_INITIAL_SOURCE_MISSING: select an existing frozen initial source')
    policy = source / 'prompts/downstream_instructions.txt'
    try:
        text = policy.read_text(encoding='utf-8')
    except OSError:
        raise SearchConfigurationError('V2_INITIAL_SOURCE_POLICY_MISSING: initial source requires readable prompts/downstream_instructions.txt') from None
    except UnicodeError:
        raise SearchConfigurationError('V2_INITIAL_SOURCE_POLICY_INVALID: downstream policy must be UTF-8') from None
    if not text.strip():
        raise SearchConfigurationError('V2_INITIAL_SOURCE_POLICY_INVALID: downstream policy must not be empty')
    if not (source / 'coding_agent.py').is_file():
        raise SearchConfigurationError('V2_INITIAL_SOURCE_AGENT_MISSING: initial source requires coding_agent.py')
    if config.wgm_v2.initial_policy_sha256 is not None and file_hash(policy) != config.wgm_v2.initial_policy_sha256:
        raise SearchConfigurationError('V2_INITIAL_POLICY_DIGEST_MISMATCH: frozen comparison policy changed')
    if config.wgm_v2.initial_source_sha256 is not None and source_snapshot(source)['sha256'] != config.wgm_v2.initial_source_sha256:
        raise SearchConfigurationError('V2_INITIAL_SOURCE_DIGEST_MISMATCH: frozen comparison source changed')
