"""Fixed branches, EXP3 and delayed rewards. No model, filesystem or evaluator IO."""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from decimal import Decimal, localcontext
import math

from search_config import WGMConfig


def lineage(history):
    counts = Counter(history)
    total = sum(counts.values())
    return total * total / sum(n*n for n in counts.values()) if total else None


class FeatureExtractor:
    """Future fixed-schema extractor; deliberately unconfigured in v0.1."""
    schema_id = None

    def extract(self, candidate):
        raise NotImplementedError("feature_schema_not_configured")


class WGMSelector:
    def __init__(self, config, rng):
        self.config = config if isinstance(config, WGMConfig) else WGMConfig(**config)
        self.rng = rng
        self.branches = {0: 0}  # node id -> immutable branch id
        self.frozen = False
        self.log_weights = {}
        self.feedback = {}
        self.parent_selections = []

    def admit(self, node_id, parent_id, *, initial=False):
        if node_id in self.branches or parent_id not in self.branches:
            raise ValueError("Duplicate node or unknown primary parent")
        if initial and self.frozen:
            raise ValueError("Initial branches are already frozen")
        branch = self.branches[parent_id]
        if initial:
            if parent_id != 0:
                raise ValueError("Initial branch seeds must be direct root children")
            count = len(set(self.branches.values()))
            if count < min(6, self.config.num_branches):
                branch = count
        self.branches[node_id] = branch
        return branch

    def freeze(self):
        if self.frozen:
            return
        ids = sorted(set(self.branches.values()))
        if len(ids) < 2:
            raise ValueError("WGM requires at least two effective branches")
        self.log_weights = {b: "0" for b in ids}
        self.frozen = True

    @property
    def effective_eta(self):
        if not self.frozen:
            raise ValueError("Branches are not frozen")
        return self.config.eta if self.config.eta is not None else self.config.gamma / len(self.log_weights)

    def snapshot(self):
        return {**asdict(self.config), "effective_branch_count": len(self.log_weights),
                "effective_eta": self.effective_eta}

    def probabilities(self, eligible):
        eligible = sorted(set(eligible))
        if not self.frozen or not eligible or any(b not in self.log_weights for b in eligible):
            raise ValueError("No eligible branches or unknown/floating branch set")
        weights = [Decimal(self.log_weights[b]) for b in eligible]
        largest = max(weights)
        scaled = [0.0 if w-largest < -745 else math.exp(float(w-largest)) for w in weights]
        total = math.fsum(scaled)
        floor = self.config.gamma / len(eligible)
        if floor == 0:
            raise ValueError("gamma exploration floor is below floating-point resolution")
        return {b: (1-self.config.gamma)*v/total + floor for b,v in zip(eligible, scaled)}

    def select(self, eligible):
        probabilities = self.probabilities(eligible)
        ids = list(probabilities)
        branch = int(self.rng.choice(ids, p=list(probabilities.values())))
        return branch, probabilities[branch]

    def begin(self, expansion_id, branch, parent_id, candidate_id, probability):
        if expansion_id in self.feedback or any(f["candidate_id"] == candidate_id for f in self.feedback.values()):
            raise ValueError("Duplicate expansion or candidate")
        if self.branches.get(parent_id) != branch or not math.isfinite(probability) or not 0 < probability <= 1:
            raise ValueError("Invalid selected branch/parent/probability")
        self.feedback[expansion_id] = {
            "expansion_id": expansion_id, "selected_branch_id": branch, "parent_id": parent_id,
            "candidate_id": candidate_id, "child_id": None, "selection_probability": probability,
            "config": self.snapshot(), "target_M": self.config.reward_batch_size,
            "results": {}, "updated": False, "reward": None, "importance_weighted_reward": None,
        }
        self.parent_selections.append(branch)

    def _complete(self, record, reward):
        if record["updated"]:
            return False
        if not math.isfinite(reward) or not 0 <= reward <= 1:
            raise ValueError("Reward must be in [0,1]")
        branch = record["selected_branch_id"]
        # Decimal log weights allow even a finite eta near float max without
        # overflow. Exponentiation is only used on non-positive differences.
        with localcontext() as context:
            context.prec = 80
            importance = Decimal(str(reward)) / Decimal(str(record["selection_probability"]))
            increment = Decimal(str(record["config"]["effective_eta"])) * importance
            self.log_weights[branch] = str(Decimal(self.log_weights[branch]) + increment)
        record.update(updated=True, reward=reward, importance_weighted_reward=str(importance))
        return True

    def invalid(self, expansion_id):
        return self._complete(self.feedback[expansion_id], 0.0)

    def observe(self, candidate_id, task_id, success):
        if type(success) not in (int, bool) or success not in (0, 1):
            raise ValueError("Task outcome must be binary")
        record = next((r for r in self.feedback.values() if r["candidate_id"] == candidate_id), None)
        if record is None or record["updated"] or task_id in record["results"]:
            return False
        record["results"][task_id] = int(success)
        if len(record["results"]) == record["target_M"]:
            return self._complete(record, sum(record["results"].values()) / record["target_M"])
        return False

    def summary(self):
        records = list(self.feedback.values())
        return {"completed": sum(r["updated"] for r in records),
                "pending": sum(not r["updated"] for r in records),
                "zero_reward": sum(r["updated"] and r["reward"] == 0 for r in records),
                "reproductive_lineage_L": lineage(self.parent_selections),
                "design_diversity_D": None, "design_diversity_reason": "feature_schema_not_configured"}

    def state_dict(self):
        name, keys, pos, gaussian, cached = self.rng.get_state()
        return deepcopy({"config": asdict(self.config), "branches": self.branches, "frozen": self.frozen,
                         "log_weights": self.log_weights, "feedback": self.feedback,
                         "parent_selections": self.parent_selections,
                         "rng": [name, keys.tolist(), pos, gaussian, cached]})

    @classmethod
    def restore(cls, state, config, rng):
        obj = cls(config, rng)
        if state["config"] != asdict(obj.config):
            raise ValueError("Resume WGM configuration differs")
        obj.branches = {int(k): v for k,v in state["branches"].items()}
        obj.log_weights = {int(k): v for k,v in state["log_weights"].items()}
        obj.frozen = state["frozen"]
        obj.feedback = deepcopy(state["feedback"])
        obj.parent_selections = list(state["parent_selections"])
        if obj.frozen and (len(obj.log_weights) < 2 or set(obj.log_weights) != set(obj.branches.values())):
            raise ValueError("Invalid saved fixed branches")
        if any(not Decimal(v).is_finite() for v in obj.log_weights.values()):
            raise ValueError("Invalid saved EXP3 weights")
        obj.rng.set_state(tuple(state["rng"]))
        return obj
