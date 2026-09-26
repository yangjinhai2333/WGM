"""Pure WGM V2 EXP3 and anytime Hoeffding BAI; no HGM/V1 statistics."""
from copy import deepcopy
import hashlib
import json
import math
import random

from wgm_v2_config import SPEC


def seed_for(domain, seed):
    raw = json.dumps([SPEC, domain, seed], ensure_ascii=True, separators=(',', ':')).encode()
    return int.from_bytes(hashlib.sha256(raw).digest(), 'big')


def rng_snapshot(rng):
    return json.loads(json.dumps(rng.getstate(), allow_nan=False))


def validate_python_rng(row):
    if not (type(row) is list and len(row) == 3 and type(row[0]) is int and row[0] == 3
            and type(row[1]) is list and len(row[1]) == 625
            and all(type(v) is int and 0 <= v < 2**32 for v in row[1][:624])
            and type(row[1][624]) is int and 0 <= row[1][624] <= 624
            and (row[2] is None or type(row[2]) is float and math.isfinite(row[2]))):
        raise ValueError('V2_RNG_SCHEMA')


def restore_rng(row):
    validate_python_rng(row)
    rng = random.Random(0)
    rng.setstate((row[0], tuple(row[1]), row[2]))
    return rng


def interval(index, n, successes, delta):
    if (type(index) is not int or index < 1 or type(n) is not int or n < 0
            or type(successes) is not int or not 0 <= successes <= n
            or type(delta) not in (float, int) or not 0 < delta < 1):
        raise ValueError('V2_BAI_COUNTS')
    if n == 0:
        return {'n': 0, 'successes': 0, 'mean': None, 'lower': 0.0, 'upper': 1.0}
    radius = math.sqrt((math.log(2 * math.pi**4 / (36 * delta))
                        + 2 * math.log(index) + 2 * math.log(n)) / (2*n))
    mean = successes / n
    return {'n': n, 'successes': successes, 'mean': mean,
            'lower': max(0.0, mean-radius), 'upper': min(1.0, mean+radius)}


def bai(rows, epsilon):
    """Rows carry candidate_id and their own interval. All outputs are copies."""
    if not rows:
        raise ValueError('V2_NO_ELIGIBLE_AGENT')
    key = lambda r: r['candidate_id'].encode('utf-8')
    leader = min(rows, key=lambda r: (-r['lower'], key(r)))
    others = [r for r in rows if r['candidate_id'] != leader['candidate_id']]
    challenger = min(others, key=lambda r: (-r['upper'], key(r))) if others else None
    reason = 'single_candidate' if challenger is None else (
        'epsilon_satisfied' if challenger['upper']-leader['lower'] <= epsilon else None)
    choice = None if reason else min([leader, challenger], key=lambda r: (
        -(r['upper']-r['lower']), r['n'], key(r)))
    return deepcopy({'leader': leader, 'challenger': challenger, 'evaluate': choice, 'stop_reason': reason})


class LineageEXP3:
    def __init__(self, founders, config, rng):
        self.founders = dict(sorted((int(k), v) for k, v in founders.items()))
        if list(self.founders) != list(range(len(self.founders))) or len(self.founders) < 2:
            raise ValueError('V2_LINEAGES_REQUIRE_TWO')
        self.config, self.rng = config, rng
        self.eta = config.eta if config.eta is not None else config.gamma / len(self.founders)
        if config.gamma / len(self.founders) <= 0:
            raise ValueError('V2_EXPLORATION_UNDERFLOW')
        self.log_weights = {c: 0.0 for c in self.founders}
        self.selections = {c: 0 for c in self.founders}
        self.feedback_counts = {c: 0 for c in self.founders}
        self.draws, self.rewards = {}, {}

    def probabilities(self):
        largest = max(self.log_weights.values())
        weights = {c: math.exp(w-largest) for c, w in self.log_weights.items()}
        total = math.fsum(weights.values())
        return {c: (1-self.config.gamma)*weights[c]/total + self.config.gamma/len(weights) for c in weights}

    def select(self, expansion_id):
        if expansion_id in self.draws:
            raise ValueError('V2_DUPLICATE_SELECTION')
        p, u = self.probabilities(), self.rng.random()
        cumulative, chosen = 0.0, next(reversed(p))
        for c, prob in p.items():
            cumulative += prob
            if u < cumulative:
                chosen = c
                break
        row = {'lineage_id': chosen, 'probability': p[chosen],
               'probabilities': {str(c): v for c, v in p.items()}, 'draw': u}
        self.draws[expansion_id] = row
        self.selections[chosen] += 1
        return deepcopy(row)

    def feedback(self, expansion_id, reward):
        if type(reward) not in (int, float) or not math.isfinite(reward) or not 0 <= reward <= 1:
            raise ValueError('V2_REWARD_INVALID')
        if expansion_id in self.rewards:
            if self.rewards[expansion_id]['reward'] != reward:
                raise ValueError('V2_REWARD_CONFLICT')
            return
        draw = self.draws[expansion_id]
        c, p = draw['lineage_id'], draw['probability']
        estimate = reward/p
        updated = self.log_weights[c] + self.eta*estimate
        if not math.isfinite(updated):
            raise ValueError('V2_WEIGHT_OVERFLOW')
        self.log_weights[c] = updated
        self.feedback_counts[c] += 1
        self.rewards[expansion_id] = {'reward': reward, 'importance_weighted_reward': estimate,
                                      'probability': p, 'eta': self.eta}

    def state_dict(self):
        return deepcopy({'schema_version': 1, 'founders': {str(k): v for k, v in self.founders.items()},
            'log_weights': {str(k): v for k, v in self.log_weights.items()},
            'probabilities': {str(k): v for k, v in self.probabilities().items()},
            'selections': {str(k): v for k, v in self.selections.items()},
            'feedback_counts': {str(k): v for k, v in self.feedback_counts.items()},
            'draws': self.draws, 'rewards': self.rewards, 'eta': self.eta, 'rng': rng_snapshot(self.rng)})

    @classmethod
    def restore(cls, row, config):
        if type(row.get('schema_version')) is not int or row['schema_version'] != 1:
            raise ValueError('V2_SELECTOR_SCHEMA')
        result = cls(row['founders'], config, restore_rng(row['rng']))
        for field in ('log_weights', 'selections', 'feedback_counts'):
            values = row[field]
            if set(values) != set(row['founders']):
                raise ValueError('V2_SELECTOR_LINEAGES')
            for value in values.values():
                if field == 'log_weights':
                    valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
                else:
                    valid = type(value) is int and value >= 0
                if not valid:
                    raise ValueError('V2_SELECTOR_VALUE')
            setattr(result, field, {int(k): v for k, v in values.items()})
        result.draws, result.rewards = deepcopy(row['draws']), deepcopy(row['rewards'])
        if result.state_dict() != row:
            raise ValueError('V2_SELECTOR_IDENTITY')
        return result
