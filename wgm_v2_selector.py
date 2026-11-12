"""WGM parent selection and BAI as in Algorithm 1: agent-level EXP3 with clade statistics,
Wilson-interval leader/challenger BAI."""
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


def interval(n, successes, z):
    """Wilson score interval (Appendix C, Eq. C.1); [0, 1] for an unevaluated agent."""
    if (type(n) is not int or n < 0 or type(successes) is not int or not 0 <= successes <= n
            or type(z) not in (float, int) or not math.isfinite(z) or z <= 0):
        raise ValueError('V2_BAI_COUNTS')
    if n == 0:
        return {'n': 0, 'successes': 0, 'mean': None, 'lower': 0.0, 'upper': 1.0}
    mean = successes / n
    centre = mean + z*z/(2*n)
    radius = z*math.sqrt(mean*(1-mean)/n + z*z/(4*n*n))
    scale = 1 + z*z/n
    return {'n': n, 'successes': successes, 'mean': mean,
            'lower': max(0.0, (centre-radius)/scale), 'upper': min(1.0, (centre+radius)/scale)}


def _argmax(rows, value, tie_key):
    """Maximum of value(row); ties are broken uniformly at random by a pure keyed draw."""
    best = max(value(r) for r in rows)
    tied = sorted((r for r in rows if value(r) == best), key=lambda r: r['candidate_id'].encode('utf-8'))
    if len(tied) == 1:
        return tied[0]
    return tied[random.Random(seed_for('bai-tie', [tie_key, value.__name__])).randrange(len(tied))]


def bai(rows, tie_key):
    """One BAISTEP decision (Eqs. 11-12): leader h = argmax lower, challenger k = argmax upper
    over A minus h, evaluate the wider interval of {h, k}; h when |A| = 1. No early stopping."""
    if not rows:
        raise ValueError('V2_NO_ELIGIBLE_AGENT')
    def lower(r): return r['lower']
    def upper(r): return r['upper']
    def width(r): return r['upper'] - r['lower']
    leader = _argmax(rows, lower, tie_key)
    others = [r for r in rows if r['candidate_id'] != leader['candidate_id']]
    challenger = _argmax(others, upper, tie_key) if others else None
    choice = leader if challenger is None else _argmax([leader, challenger], width, tie_key)
    return deepcopy({'leader': leader, 'challenger': challenger, 'evaluate': choice, 'stop_reason': None})


class AgentEXP3:
    """EXP3 parent selection over all archived agents (Eqs. 6-8, 10).

    pi_a = (1-gamma) w_a rho_a / sum_u w_u rho_u + gamma/N_t, eta_t = gamma/N_t, and only the
    selected parent receives w <- w exp(eta_t r / pi_a). New agents enter with weight one."""
    def __init__(self, agents, config, rng):
        self.config, self.rng = config, rng
        self.log_weights = {}
        self.selections, self.feedback_counts = {}, {}
        self.draws, self.rewards = {}, {}
        for agent in agents:
            self.add(agent)

    def add(self, agent):
        if type(agent) is not str or agent in self.log_weights:
            raise ValueError('V2_AGENT_IDENTITY')
        self.log_weights[agent] = 0.0
        self.selections[agent] = 0
        self.feedback_counts[agent] = 0

    def probabilities(self, rho):
        if set(rho) != set(self.log_weights) or not rho:
            raise ValueError('V2_CLADE_STATISTICS')
        gamma, n = self.config.gamma, len(self.log_weights)
        largest = max(self.log_weights.values())
        scores = {a: math.exp(w-largest) * rho[a] for a, w in self.log_weights.items()}
        total = math.fsum(scores.values())
        # min() only absorbs float rounding (e.g. 1.0000000000000002 when N_t = 1).
        return {a: min(1.0, (1-gamma)*scores[a]/total + gamma/n) for a in sorted(scores)}

    def select(self, expansion_id, rho):
        if expansion_id in self.draws:
            raise ValueError('V2_DUPLICATE_SELECTION')
        p, u = self.probabilities(rho), self.rng.random()
        cumulative, chosen = 0.0, next(reversed(p))
        for a, prob in p.items():
            cumulative += prob
            if u < cumulative:
                chosen = a
                break
        row = {'candidate_id': chosen, 'probability': p[chosen], 'probabilities': p, 'draw': u,
               'eta': self.config.gamma / len(p), 'archive_size': len(p)}
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
        a, p, eta = draw['candidate_id'], draw['probability'], draw['eta']
        estimate = reward/p
        updated = self.log_weights[a] + eta*estimate
        if not math.isfinite(updated):
            raise ValueError('V2_WEIGHT_OVERFLOW')
        self.log_weights[a] = updated
        self.feedback_counts[a] += 1
        self.rewards[expansion_id] = {'reward': reward, 'importance_weighted_reward': estimate,
                                      'probability': p, 'eta': eta}

    def state_dict(self):
        return deepcopy({'schema_version': 2, 'agents': list(self.log_weights),
            'log_weights': self.log_weights, 'selections': self.selections,
            'feedback_counts': self.feedback_counts, 'draws': self.draws, 'rewards': self.rewards,
            'rng': rng_snapshot(self.rng)})

    @classmethod
    def restore(cls, row, config):
        if type(row.get('schema_version')) is not int or row['schema_version'] != 2:
            raise ValueError('V2_SELECTOR_SCHEMA')
        result = cls(row['agents'], config, restore_rng(row['rng']))
        for field in ('log_weights', 'selections', 'feedback_counts'):
            values = row[field]
            if set(values) != set(row['agents']) or len(set(row['agents'])) != len(row['agents']):
                raise ValueError('V2_SELECTOR_AGENTS')
            for value in values.values():
                if field == 'log_weights':
                    valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
                else:
                    valid = type(value) is int and value >= 0
                if not valid:
                    raise ValueError('V2_SELECTOR_VALUE')
            setattr(result, field, {a: values[a] for a in row['agents']})
        result.draws, result.rewards = deepcopy(row['draws']), deepcopy(row['rewards'])
        if result.state_dict() != row:
            raise ValueError('V2_SELECTOR_IDENTITY')
        return result
