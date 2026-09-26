"""Opt-in execution controls for the unmodified serial legacy HGM trajectory."""
from dataclasses import dataclass

from search_config import SearchConfigurationError


@dataclass(frozen=True)
class HGMIncrementalConfig:
    planning_horizon: int = 800
    target_budget: int = 100
    checkpoints: tuple = (100, 200, 400, 800)

    def __post_init__(self):
        if type(self.planning_horizon) is not int or self.planning_horizon != 800:
            raise SearchConfigurationError("Incremental HGM planning_horizon must remain 800")
        if type(self.target_budget) is not int or not 0 < self.target_budget <= self.planning_horizon:
            raise SearchConfigurationError("HGM target must be in [1, planning_horizon]")
        if list(self.checkpoints) != [100, 200, 400, 800]:
            raise SearchConfigurationError("HGM checkpoints must be [100, 200, 400, 800]")

    def identity(self):
        return {"schema_version": 1, "planning_horizon": self.planning_horizon,
                "checkpoints": list(self.checkpoints)}


def enabled(config):
    return bool(getattr(config, "hgm_incremental", None))


def validate(config):
    if not enabled(config):
        return
    if not isinstance(config.hgm_incremental, HGMIncrementalConfig):
        try:
            config.hgm_incremental = HGMIncrementalConfig(**config.hgm_incremental)
        except TypeError as error:
            raise SearchConfigurationError("Invalid HGM incremental configuration") from error
    if (config.search.algorithm != "hgm" or config.search.hgm_profile != "legacy_code"
            or config.execution.policy != "validated" or config.optimization.cool_down):
        raise SearchConfigurationError("Incremental HGM requires validated legacy_code with cool_down=false")
