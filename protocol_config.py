"""Explicit opt-in experiment controls; legacy search ignores these settings."""
from dataclasses import dataclass


class ProtocolConfigurationError(ValueError):
    pass


@dataclass
class ProtocolConfig:
    name: str | None = None
    search_task_evals: int | None = None
    smoke: bool = False
    allow_legacy_initial_cache: bool = False
    final_eval_task_limit: int | None = None

    @property
    def effective_name(self):
        return self.name or ("smoke" if self.smoke else "formal")

    def validate(self):
        if self.name is not None and self.name not in ("formal", "pilot", "smoke"):
            raise ProtocolConfigurationError("protocol must be formal, pilot or smoke")
        if type(self.smoke) is not bool or type(self.allow_legacy_initial_cache) is not bool:
            raise ProtocolConfigurationError("Protocol flags must be booleans")
        if self.smoke and self.name not in (None, "smoke"):
            raise ProtocolConfigurationError("--smoke cannot override an explicit formal/pilot protocol")
        if self.allow_legacy_initial_cache and self.effective_name != "smoke":
            raise ProtocolConfigurationError("legacy_unverified_cache requires explicit smoke mode")
        budget = self.search_task_evals
        if budget is not None and (type(budget) is not int or budget < 1):
            raise ProtocolConfigurationError("search_task_evals must be a positive integer")
        n = self.final_eval_task_limit
        if n is not None and (type(n) is not int or n < 1):
            raise ProtocolConfigurationError("final_eval_task_limit must be a positive integer")
        if self.effective_name != "smoke" and n is not None and n != 60:
            raise ProtocolConfigurationError("Formal/pilot final evaluation requires all 60 tasks")

    def search_budget(self, configured_budget):
        return (self.search_task_evals if self.search_task_evals is not None else
                20 if self.effective_name == "pilot" else configured_budget)

    def identity(self, configured_budget):
        """Effective values, independent of equivalent CLI/YAML spellings."""
        return {"name": self.effective_name,
                "search_task_evals": self.search_budget(configured_budget),
                "final_evaluation_tasks": self.final_eval_task_limit or 60,
                "allow_legacy_initial_cache": self.allow_legacy_initial_cache}


def configure_search_protocol(config, explicit_budget=None):
    """Resolve protocol at the entry boundary, before preflight or model calls."""
    from search_config import is_corrected
    protocol = config.protocol
    protocol.validate()
    from hgm_incremental_config import enabled, validate
    if enabled(config):
        validate(config)
        if explicit_budget is not None:
            raise ProtocolConfigurationError("Incremental HGM uses --target-budget, not --max_task_evals")
        config.execution.max_task_evals = config.hgm_incremental.planning_horizon
        return
    if config.search.algorithm == "dgm":
        if explicit_budget is not None:
            raise ProtocolConfigurationError("DGM uses --target-budget, not --max_task_evals")
        config.execution.max_task_evals = config.dgm.planning_horizon
        return
    if config.search.algorithm == "wgm_v2":
        if explicit_budget is not None:
            raise ProtocolConfigurationError("WGM V2 uses --target-budget, not --max_task_evals")
        config.execution.max_task_evals = config.wgm_v2.planning_horizon
        return
    if not is_corrected(config):
        if protocol.effective_name == "pilot":
            raise ProtocolConfigurationError("pilot requires HGM paper_corrected or WGM")
        return
    if explicit_budget is not None:
        protocol.search_task_evals = explicit_budget
    budget = protocol.search_budget(config.execution.max_task_evals)
    if type(budget) is not int or budget < 1:
        raise ProtocolConfigurationError("Search budget must be a positive integer")
    if protocol.effective_name == "formal" and budget != 800:
        raise ProtocolConfigurationError("Formal protocol requires 800 search evaluations; use --protocol pilot for a lower budget")
    config.execution.max_task_evals = budget


@dataclass
class DiagnosisContextConfig:
    max_input_chars: int = 48000
    fallback_input_chars: int = 24000
    min_input_chars: int = 6000
    strategy: str = "structured_head_tail_v1"

    def validate(self):
        values = (self.max_input_chars, self.fallback_input_chars, self.min_input_chars)
        if any(type(n) is not int for n in values) or not (4096 <= values[2] < values[1] < values[0]):
            raise ProtocolConfigurationError("Diagnosis budgets must satisfy 4096 <= min < fallback < max")
        if self.strategy != "structured_head_tail_v1":
            raise ProtocolConfigurationError("Unknown diagnosis context strategy")
