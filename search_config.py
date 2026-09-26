"""Search identity and effective profile settings; no execution backends."""
from dataclasses import asdict, dataclass, replace
import math


class SearchConfigurationError(ValueError):
    pass


@dataclass
class SearchConfig:
    algorithm: str = "hgm"
    hgm_profile: str = "legacy_code"
    seed: int = 42
    dgm_profile: str = "legacy_code"

    def validate(self):
        if self.algorithm not in ("hgm", "wgm", "wgm_v2", "dgm"):
            raise SearchConfigurationError("search.algorithm must be hgm, wgm, wgm_v2 or dgm")
        if self.algorithm == "dgm" and self.dgm_profile != "legacy_code":
            raise SearchConfigurationError("Only dgm_profile=legacy_code is supported")
        if self.hgm_profile not in ("legacy_code", "paper_corrected"):
            raise SearchConfigurationError("Unknown hgm_profile")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise SearchConfigurationError("search.seed must be an integer in [0, 2**32)")


@dataclass(frozen=True)
class WGMConfig:
    num_branches: int = 6
    gamma: float = 0.1
    reward_batch_size: int = 1
    eta: float | None = None
    feedback_mode: str = "delayed"
    final_selection: str = "hgm_best_belief"

    def __post_init__(self):
        for name, lower in (("num_branches", 2), ("reward_batch_size", 1)):
            value = getattr(self, name)
            if type(value) is not int or value < lower:
                raise SearchConfigurationError(f"wgm.{name} must be an integer >= {lower}")
        for name, value in (("gamma", self.gamma), ("eta", self.eta)):
            if value is None and name == "eta":
                continue
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0):
                raise SearchConfigurationError(f"wgm.{name} must be finite and positive")
        if self.gamma > 1:
            raise SearchConfigurationError("wgm.gamma must be <= 1")
        if self.feedback_mode != "delayed" or self.final_selection != "hgm_best_belief":
            raise SearchConfigurationError("Only delayed feedback and hgm_best_belief are supported")


def is_corrected(config):
    return config.search.algorithm in ("wgm", "wgm_v2") or config.search.hgm_profile == "paper_corrected"


def optimization(config):
    return (replace(config.optimization, alpha=0.6, cool_down=True, beta=1.0)
            if is_corrected(config) else config.optimization)


def experiment(config, run_type, effective_branch_count=None):
    if run_type == "dgm":
        from dgm_legacy_config import identity
        return identity(config)
    if run_type == "wgm_v2":
        from wgm_v2_config import identity
        return identity(config)
    if run_type not in ("hgm", "wgm"):
        return {"algorithm": "baseline"}
    config.search.validate()
    corrected = is_corrected(config)
    value = {"algorithm": config.search.algorithm}
    value["shared_hgm_profile" if config.search.algorithm == "wgm" else "hgm_profile"] = (
        "paper_corrected" if corrected else "legacy_code")
    value["optimization"] = asdict(optimization(config))
    value["initial_expansion_target"] = 5 if corrected else min(
        5, int(config.execution.max_workers ** config.optimization.alpha))
    value["seed"] = config.search.seed if corrected else None
    if corrected:
        value["spec_version"] = "WGM v0.1 minimal-delta"
        value["protocol"] = config.protocol.identity(config.execution.max_task_evals)
        value["search_budget"] = value["protocol"]["search_task_evals"]
        value["diagnosis_context"] = asdict(config.diagnosis_context)
    value["rng"] = "saved_python_and_numpy" if corrected else "legacy_global_rng"
    from hgm_incremental_config import enabled
    if enabled(config):
        value["incremental"] = config.hgm_incremental.identity()
        value["seed"] = config.search.seed
    value["final_selection"] = ({"method": "hgm_best_belief", "epsilon_percent": 1.0,
                                 "q": 0.01, "convention": "provisional_lower_quantile"}
                                if corrected else None)
    if config.search.algorithm == "wgm":
        cfg = config.wgm
        if not isinstance(cfg, WGMConfig):
            cfg = WGMConfig(**cfg)
        value["wgm"] = asdict(cfg)
        value["wgm"]["effective_branch_count"] = effective_branch_count
        value["wgm"]["effective_eta"] = (cfg.eta if cfg.eta is not None else
                                         cfg.gamma / effective_branch_count if effective_branch_count else None)
    return value
