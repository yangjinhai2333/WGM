"""Canonical task pool; scheduling within this pool remains the search's job."""
import hashlib
import json
from pathlib import Path
import random

CANONICAL_MEMBERSHIP_HASH = "b151af816c3d2a9f8d328b9af6991bdb9937f760bb64122aca1c2b0738a23e30"
# Polyglot small10 + medium50 share one ID: 59 distinct tasks, order seed 42.
POLYGLOT_MEMBERSHIP_HASH = "820fac4f1ef05a2fceb2f35af6d81f07ce38650c989482d4fd49f518c7baf8f2"
BENCHMARKS = {
    False: {"name": "swe", "count": 60, "membership_hash": CANONICAL_MEMBERSHIP_HASH,
            "order_hash_42": "bcc4567b75cb219a43e12fbd4a89cfe38c81976d11f826ac78b7a1447d032bc1"},
    True: {"name": "polyglot", "count": 59, "membership_hash": POLYGLOT_MEMBERSHIP_HASH,
           "order_hash_42": "f84c2139712a458a29da80a760153f9a3c899c88f29d3c93748eeb520cce3f5f"},
}


def benchmark(polyglot=False):
    """Frozen identity of the canonical small+medium pool of one benchmark."""
    return BENCHMARKS[bool(polyglot)]


def pool_size(polyglot=False):
    return benchmark(polyglot)["count"]


def count_for_membership(membership_hash):
    """Pool size of a frozen canonical membership; the historical 60 for any other.

    Before Polyglot support every compared final was required to have exactly 60
    tasks, whatever its membership hash; that rule is kept for unknown sets.
    """
    return next((b["count"] for b in BENCHMARKS.values() if b["membership_hash"] == membership_hash), 60)


def manifest_polyglot(manifest):
    """Benchmark of a recorded run (SWE when the field predates Polyglot support)."""
    return bool(manifest["configuration"]["effective"].get("evaluation", {}).get("polyglot"))


def task_hash(ids):
    return hashlib.sha256(json.dumps(list(ids), ensure_ascii=True,
                                    separators=(",", ":")).encode()).hexdigest()


def unique_tasks(ids):
    ids = list(ids)
    if any(not isinstance(t, str) or not t or Path(t).name != t or t in (".", "..") for t in ids):
        raise ValueError("Invalid task ID")
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate task IDs")
    return ids


def ordered_tasks(ids, seed=42):
    """Shuffle the checked-in small+medium order using an isolated Python RNG."""
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("task_order_seed must be an integer in [0, 2**32)")
    ids = unique_tasks(ids)
    random.Random(seed).shuffle(ids)
    return ids


def canonical_tasks(root, seed=42, *, polyglot=False):
    folder = Path(root) / ("polyglot" if polyglot else "swe_bench") / "subsets"
    ids = json.loads((folder / "small.json").read_text()) + json.loads((folder / "medium.json").read_text())
    if polyglot:
        ids = list(dict.fromkeys(ids))  # The subsets overlap by one task.
    return ordered_tasks(ids, seed)


def schedule_identity(ids, seed):
    ids = unique_tasks(ids)
    return {"task_membership_hash": task_hash(sorted(ids)), "task_order_hash": task_hash(ids),
            "task_order_seed": seed, "schedule_schema": "canonical_task_schedule_v1"}


def validate_schedule(tasks):
    if tasks.get("schedule_schema") != "canonical_task_schedule_v1":
        raise ValueError("Canonical task schedule is legacy/unavailable")
    ordered_tasks([], tasks.get("task_order_seed"))
    expected = schedule_identity(tasks["ordered_ids"], tasks["task_order_seed"])
    if any(tasks.get(k) != v for k, v in expected.items()) or tasks.get("count") != len(tasks["ordered_ids"]):
        raise ValueError("Task schedule hashes/count disagree")
    return expected
