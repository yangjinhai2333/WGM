"""
This module provides functionality to load and manage configuration parameters
from YAML files with fallback to default values.
"""

import os
import yaml
from dataclasses import asdict, dataclass, field
from typing import Optional, List, Dict, Any

from runners.codex_exec import CodexExecConfig, CodexConfigurationError, validate_reasoning_effort
from runners.claude_code import ClaudeCodeConfig, validate_claude_effort
from candidate_workspace import CandidateConfig
from search_config import SearchConfig, WGMConfig, SearchConfigurationError
from wgm_v2_config import WGMV2Config
from protocol_config import ProtocolConfig, DiagnosisContextConfig, ProtocolConfigurationError
from execution_policy import ExecutionPolicyError, validate_name as validate_execution_policy
from network_readiness import ReadinessConfig, ReadinessConfigurationError


class InvalidDiagnosisBackend(ValueError):
    """An unsupported diagnosis backend must not silently fall back to defaults."""


class InvalidSelfImproveBackend(ValueError):
    """An unsupported self-improvement backend is a startup error."""


class InvalidDownstreamBackend(ValueError):
    """An unsupported downstream backend is a startup error."""


class EvaluationWorkersError(ValueError):
    """Invalid parallelism must never silently fall back to a default."""


def validate_evaluation_workers(value):
    if type(value) is not int or value < 1:
        raise EvaluationWorkersError("execution.evaluation_workers must be an integer >= 1")
    return value


def validate_downstream_backend(backend):
    if backend not in ("codex_exec", "legacy_api"):
        raise InvalidDownstreamBackend(
            "llm.downstream_backend must be 'codex_exec' or 'legacy_api'."
        )


def validate_self_improve_backend(backend):
    if backend not in ("codex_exec", "legacy_api"):
        raise InvalidSelfImproveBackend(
            "llm.self_improve_backend must be 'codex_exec' or 'legacy_api'."
        )


def effective_worker_count(configured_workers, self_improve_backend, downstream_backend="legacy_api",
                           diagnose_backend="legacy_api"):
    validate_self_improve_backend(self_improve_backend)
    validate_downstream_backend(downstream_backend)
    validate_diagnose_backend(diagnose_backend)
    return 1 if "codex_exec" in (self_improve_backend, downstream_backend, diagnose_backend) else configured_workers


def validate_diagnose_backend(backend):
    if backend not in ("codex_exec", "legacy_api"):
        raise InvalidDiagnosisBackend(
            "llm.diagnose_backend must be 'codex_exec' or 'legacy_api'."
        )


def codex_config_from_dict(data):
    """Build the agent-CLI config; ``provider`` selects Codex (default) or Claude."""
    data = dict(data or {})
    provider = data.pop('provider', 'codex')
    if provider == 'codex':
        config = CodexExecConfig(**data)
        validate_reasoning_effort(config.reasoning_effort)
        return config
    if provider == 'claude_code':
        return ClaudeCodeConfig(**data)
    raise CodexConfigurationError("codex.provider must be 'codex' or 'claude_code'.")


@dataclass
class LLMConfig:
    """Configuration for Language Model settings."""
    self_improve_llm: str = "gpt-5-mini"
    downstream_llm: str = "gpt-5-mini"
    diagnose_llm: str = "gpt-5-mini"
    diagnose_backend: str = "codex_exec"
    self_improve_backend: str = "codex_exec"
    downstream_backend: str = "codex_exec"

    def __post_init__(self):
        validate_diagnose_backend(self.diagnose_backend)
        validate_self_improve_backend(self.self_improve_backend)
        validate_downstream_backend(self.downstream_backend)


@dataclass
class OptimizationConfig:
    """Configuration for optimization algorithm parameters."""
    alpha: float = 0.6
    beta: float = 1.0
    cool_down: bool = False
    eval_random_level: float = 1.0
    n_pseudo_descendant_evals: int = 10000


@dataclass
class ExecutionConfig:
    """Configuration for execution and resource management."""
    max_workers: int = 20
    evaluation_workers: int = 1
    self_improve_timeout: int = 3600
    evaluation_timeout: int = 3600
    max_task_evals: int = 800
    policy: str = "origin"

    def __post_init__(self):
        validate_execution_policy(self.policy)
        validate_evaluation_workers(self.evaluation_workers)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation settings."""
    full_eval: bool = False
    polyglot: bool = False
    # Keep the original harness dataset and the checked-in subset IDs.
    dataset_name: str = "princeton-nlp/SWE-bench"
    dataset_path: str = "swe_bench/dataset"
    polyglot_dataset_path: str = "polyglot/polyglot_benchmark_metadata.json"
    task_order_seed: int = 42


@dataclass
class PathConfig:
    """Configuration for file paths and directories."""
    output_dir: Optional[str] = None
    continue_from: Optional[str] = None
    initial_agent_name: str = ""
    root_cache_dir: Optional[str] = None
    initial_baseline_dir: Optional[str] = None


@dataclass
class HGMConfig:
    """Main configuration class containing all HGM settings."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    codex: CodexExecConfig = field(default_factory=CodexExecConfig)
    candidate: CandidateConfig = field(default_factory=CandidateConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    wgm: Any = field(default_factory=WGMConfig)
    wgm_v2: Any = field(default_factory=dict)
    dgm: Any = field(default_factory=dict)
    hgm_incremental: Any = field(default_factory=dict)
    protocol: ProtocolConfig = field(default_factory=ProtocolConfig)
    diagnosis_context: DiagnosisContextConfig = field(default_factory=DiagnosisContextConfig)
    network_readiness: ReadinessConfig = field(default_factory=ReadinessConfig)

    def __post_init__(self):
        self.codex.network_readiness = self.network_readiness
        self.validate_search()

    def validate_search(self):
        validate_execution_policy(self.execution.policy)
        validate_evaluation_workers(self.execution.evaluation_workers)
        self.search.validate()
        from hgm_incremental_config import validate
        validate(self)
        seed = self.evaluation.task_order_seed
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ProtocolConfigurationError("task_order_seed must be an integer in [0, 2**32)")
        self.protocol.validate()
        self.diagnosis_context.validate()
        if self.search.algorithm == "dgm":
            from dgm_legacy_config import DGMConfig
            if not isinstance(self.dgm, DGMConfig):
                try:
                    self.dgm = DGMConfig(**self.dgm)
                except TypeError as error:
                    raise SearchConfigurationError("Invalid DGM configuration keys") from error
        if self.search.algorithm == "wgm_v2" and not isinstance(self.wgm_v2, WGMV2Config):
            try:
                self.wgm_v2 = WGMV2Config(**self.wgm_v2)
            except TypeError as error:
                raise SearchConfigurationError("Invalid WGM V2 configuration keys") from error
        if self.search.algorithm == "wgm" and not isinstance(self.wgm, WGMConfig):
            try:
                self.wgm = WGMConfig(**self.wgm)
            except TypeError as error:
                raise SearchConfigurationError("Invalid WGM configuration keys") from error

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'HGMConfig':
        """
        Load configuration from a YAML file.
        
        Args:
            yaml_path: Path to the YAML configuration file
            
        Returns:
            HGMConfig instance with loaded configuration
        """
        if not os.path.exists(yaml_path):
            print(f"Warning: Configuration file {yaml_path} not found. Using defaults.")
            return cls()
        
        execution_data = None
        config_data = None
        try:
            with open(yaml_path, 'r') as f:
                config_data = yaml.safe_load(f) or {}
            execution_data = config_data.get('execution', {})
            try:
                network_config = ReadinessConfig(**config_data.get('network_readiness', {}))
            except TypeError:
                raise ReadinessConfigurationError("NETWORK_READINESS_CONFIG_INVALID") from None
            # Reject unknown policies even when another execution key is invalid;
            # the legacy generic config fallback must not silently dispatch origin.
            if isinstance(execution_data, dict):
                validate_execution_policy(execution_data.get('policy', 'origin'))
            
            # Create instances with loaded data
            llm_config = LLMConfig(**config_data.get('llm', {}))
            optimization_config = OptimizationConfig(**config_data.get('optimization', {}))
            execution_config = ExecutionConfig(**execution_data)
            evaluation_config = EvaluationConfig(**config_data.get('evaluation', {}))
            paths_config = PathConfig(**config_data.get('paths', {}))
            codex_config = codex_config_from_dict(config_data.get('codex', {}))
            candidate_config = CandidateConfig(**config_data.get('candidate', {}))
            try:
                search_config = SearchConfig(**config_data.get('search', {}))
                search_config.validate()
            except TypeError as error:
                raise SearchConfigurationError("Invalid search configuration keys") from error
            try:
                protocol_data = config_data.get('protocol', {})
                if isinstance(protocol_data, str):
                    protocol_data = {'name': protocol_data}
                protocol_config = ProtocolConfig(**protocol_data)
                protocol_config.validate()
                # Preserve explicitly configured budgets for YAML pilot runs.
                if protocol_config.effective_name == 'pilot' and 'max_task_evals' in config_data.get('execution', {}):
                    supplied = execution_config.max_task_evals
                    if protocol_config.search_task_evals is not None and protocol_config.search_task_evals != supplied:
                        raise ProtocolConfigurationError("Conflicting protocol/execution search budgets")
                    protocol_config.search_task_evals = supplied
                context_config = DiagnosisContextConfig(**config_data.get('diagnosis_context', {}))
            except TypeError as error:
                raise ProtocolConfigurationError("Invalid protocol/context configuration keys") from error
            
            config = cls(
                llm=llm_config,
                optimization=optimization_config,
                execution=execution_config,
                evaluation=evaluation_config,
                paths=paths_config,
                codex=codex_config,
                candidate=candidate_config,
                search=search_config,
                wgm=config_data.get('wgm', {}),
                wgm_v2=config_data.get('wgm_v2', {}),
                dgm=config_data.get('dgm', {}),
                hgm_incremental=config_data.get('hgm_incremental', {}),
                protocol=protocol_config,
                diagnosis_context=context_config,
                network_readiness=network_config,
            )
            if 'policy' in config_data.get('execution', {}):
                config._execution_policy_selection = (execution_config.policy, "config")
            return config
        except (InvalidDiagnosisBackend, InvalidSelfImproveBackend, InvalidDownstreamBackend, CodexConfigurationError, SearchConfigurationError, ProtocolConfigurationError, ExecutionPolicyError, ReadinessConfigurationError, EvaluationWorkersError):
            raise
        except Exception as e:
            if isinstance(config_data, dict) and 'network_readiness' in config_data:
                # Never silently disable an explicitly configured gate because
                # another YAML section is malformed, or print private URL data.
                raise ReadinessConfigurationError("NETWORK_READINESS_CONFIG_INVALID") from None
            if isinstance(execution_data, dict) and 'policy' in execution_data:
                raise ExecutionPolicyError("EXECUTION_POLICY_CONFIGURATION_INVALID") from None
            print(f"Error loading configuration from {yaml_path}: {e}")
            print("Using default configuration.")
            return cls()
    
    @classmethod
    def from_yaml_with_overrides(cls, yaml_path: str, **overrides) -> 'HGMConfig':
        """
        Load configuration from YAML and apply command-line overrides.
        
        Args:
            yaml_path: Path to the YAML configuration file
            **overrides: Key-value pairs to override configuration values
            
        Returns:
            HGMConfig instance with loaded and overridden configuration
        """
        config = cls.from_yaml(yaml_path)
        
        # Apply overrides using dot notation (e.g., "llm.self_improve_llm")
        for key, value in overrides.items():
            if value is not None:  # Only override if value is provided
                config._set_nested_attr(key, value)
        
        config.validate_search()
        return config
    
    def _set_nested_attr(self, attr_path: str, value: Any):
        """
        Set a nested attribute using dot notation.
        
        Args:
            attr_path: Dot-separated attribute path (e.g., "llm.self_improve_llm")
            value: Value to set
        """
        if attr_path.startswith('wgm.') and isinstance(self.wgm, WGMConfig):
            self.wgm = asdict(self.wgm)
        if attr_path.startswith('wgm_v2.') and isinstance(self.wgm_v2, WGMV2Config):
            self.wgm_v2 = asdict(self.wgm_v2)
        if attr_path.startswith('dgm.') and not isinstance(self.dgm, dict):
            self.dgm = asdict(self.dgm)
        if attr_path.startswith('hgm_incremental.') and not isinstance(self.hgm_incremental, dict):
            self.hgm_incremental = asdict(self.hgm_incremental)
        parts = attr_path.split('.')
        obj = self
        
        # Navigate to the parent object
        for part in parts[:-1]:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                return  # Invalid path, skip
        
        # Set the final attribute
        final_attr = parts[-1]
        if hasattr(obj, final_attr):
            if attr_path == "execution.evaluation_workers":
                validate_evaluation_workers(value)
            if attr_path == "execution.policy":
                from execution_policy import configure
                configure(self, value)
                return
            if attr_path == "codex.reasoning_effort":
                (validate_claude_effort if isinstance(self.codex, ClaudeCodeConfig)
                 else validate_reasoning_effort)(value)
            if attr_path == "llm.diagnose_backend":
                validate_diagnose_backend(value)
            if attr_path == "llm.self_improve_backend":
                validate_self_improve_backend(value)
            if attr_path == "llm.downstream_backend":
                validate_downstream_backend(value)
            setattr(obj, final_attr, value)
        elif isinstance(obj, dict):
            obj[final_attr] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary format.
        
        Returns:
            Dictionary representation of the configuration
        """
        return {
            'network_readiness': asdict(self.network_readiness),
            'llm': {
                'self_improve_llm': self.llm.self_improve_llm,
                'downstream_llm': self.llm.downstream_llm,
                'diagnose_llm': self.llm.diagnose_llm,
                'diagnose_backend': self.llm.diagnose_backend,
                'self_improve_backend': self.llm.self_improve_backend,
                'downstream_backend': self.llm.downstream_backend,
            },
            'optimization': {
                'alpha': self.optimization.alpha,
                'beta': self.optimization.beta,
                'cool_down': self.optimization.cool_down,
                'eval_random_level': self.optimization.eval_random_level,
                'n_pseudo_descendant_evals': self.optimization.n_pseudo_descendant_evals,
            },
            'execution': {
                'policy': self.execution.policy,
                'max_workers': self.execution.max_workers,
                'evaluation_workers': self.execution.evaluation_workers,
                'self_improve_timeout': self.execution.self_improve_timeout,
                'evaluation_timeout': self.execution.evaluation_timeout,
                'max_task_evals': self.execution.max_task_evals,
            },
            'evaluation': {
                'full_eval': self.evaluation.full_eval,
                'polyglot': self.evaluation.polyglot,
                'dataset_name': self.evaluation.dataset_name,
                'dataset_path': self.evaluation.dataset_path,
                'polyglot_dataset_path': self.evaluation.polyglot_dataset_path,
                'task_order_seed': self.evaluation.task_order_seed,
            },
            'paths': {
                'output_dir': self.paths.output_dir,
                'continue_from': self.paths.continue_from,
                'initial_agent_name': self.paths.initial_agent_name,
                'root_cache_dir': self.paths.root_cache_dir,
                'initial_baseline_dir': self.paths.initial_baseline_dir,
            },
            'codex': asdict(self.codex),
            'candidate': asdict(self.candidate),
            'search': {k: v for k, v in asdict(self.search).items()
                       if k != 'dgm_profile' or self.search.algorithm == 'dgm'},
            'protocol': asdict(self.protocol),
            'diagnosis_context': asdict(self.diagnosis_context),
            **({'wgm': asdict(self.wgm) if isinstance(self.wgm, WGMConfig) else dict(self.wgm)}
               if self.search.algorithm == 'wgm' else {}),
            **({'wgm_v2': asdict(self.wgm_v2)} if self.search.algorithm == 'wgm_v2' else {}),
            **({'dgm': asdict(self.dgm)} if self.search.algorithm == 'dgm' else {}),
            **({'hgm_incremental': asdict(self.hgm_incremental)} if self.hgm_incremental else {}),
        }
    
    def save_to_yaml(self, yaml_path: str):
        """
        Save configuration to a YAML file.
        
        Args:
            yaml_path: Path where to save the YAML configuration file
        """
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)


def load_config(config_path: str = "config.yaml", **overrides) -> HGMConfig:
    """
    Convenience function to load configuration with overrides.
    
    Args:
        config_path: Path to the configuration file
        **overrides: Command-line or programmatic overrides
        
    Returns:
        HGMConfig instance
    """
    return HGMConfig.from_yaml_with_overrides(config_path, **overrides)
