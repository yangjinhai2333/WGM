"""One explicit execution policy per run; no execution or algorithm imports."""
from pathlib import Path
import json
import warnings

ORIGIN = "origin_hgm_v1"
VALIDATED = "validated_observation_v2"
NAMES = {"origin": ORIGIN, "validated": VALIDATED}


class ExecutionPolicyError(ValueError):
    pass


def validate_name(name):
    if type(name) is not str or name not in NAMES:
        raise ExecutionPolicyError("EXECUTION_POLICY_UNKNOWN")


def configure(config, requested=None):
    """Resolve CLI > configuration > origin, without dispatching either policy."""
    if requested is not None:
        validate_name(requested)
        config.execution.policy = requested
        config._execution_policy_selection = (requested, "cli")
    return record(config)


def record(config):
    """Selection provenance is diagnostic; only effective semantics identify a run."""
    requested = config.execution.policy
    selection = getattr(config, "_execution_policy_selection", None)
    # Direct programmatic edits are configuration too, never an implicit CLI opt-in.
    source = ("default" if requested == "origin" else "config") if selection is None else (
        selection[1] if selection[0] == requested else "config")
    return identity(requested, policy_source=source)


def validated(config):
    validate_name(config.execution.policy)
    return config.execution.policy == "validated"


def require_supported(config):
    if not validated(config):
        raise ExecutionPolicyError("EXECUTION_POLICY_REQUIRES_VALIDATED")
    if config.llm.downstream_backend != "codex_exec":
        raise ExecutionPolicyError("VALIDATED_BACKEND_UNSUPPORTED")


def identity(requested, *, policy_source="default", algorithm=None, profile=None, protocol=None, explicit=False):
    """Pure future V2 policy gate. It does not implement or enable a V2 selector."""
    validate_name(requested)
    if policy_source not in ("cli", "config", "default"):
        raise ExecutionPolicyError("EXECUTION_POLICY_SOURCE_UNKNOWN")
    v2 = algorithm == "wgm" and profile in ("v2", "wgm_exp3_bai_v2")
    if v2 and requested == "origin":
        if protocol == "formal":
            raise ExecutionPolicyError("WGM_V2_FORMAL_REQUIRES_VALIDATED")
        if not explicit or protocol not in ("smoke", "exploratory"):
            raise ExecutionPolicyError("WGM_V2_ORIGIN_REQUIRES_EXPLICIT_EXPLORATORY")
    good = requested == "validated"
    return {"schema_version": 1, "requested_policy": requested, "effective_policy": NAMES[requested],
            "policy_source": policy_source,
            "policy_version": 2 if good else 1,
            "empty_patch_policy": "evaluate" if good else "skip_evaluator_zero",
            "cleanup_policy": "independent_of_binding" if good else "legacy_may_override_result",
            "ledger_enabled": good, "task_recovery": "ledger_v2" if good else "legacy",
            "budget_denominator": "trusted_successful_task_evaluations" if good else "legacy_precharged_tasks",
            "failure_observation": None if good else 0,
            "adaptive_failure_policy": "stop" if good else "legacy_continue",
            "v2_failure_semantics_compatible": good,
            # A failure policy alone cannot establish IID sampling or stationarity.
            "v2_statistical_assumptions_satisfied": False,
            "statistical_guarantee": "not_validated_iid", "shadow_policy_metrics": False}


def semantics(row):
    """Validate a manifest policy and exclude its non-scientific selection source."""
    if type(row) is not dict:
        raise ExecutionPolicyError("EXECUTION_POLICY_RECORD_INVALID")
    expected = identity(row.get("requested_policy"), policy_source=row.get("policy_source"))
    # Canonical JSON distinguishes bool/int and rejects unexpected fields as well.
    if json.dumps(row, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True):
        raise ExecutionPolicyError("EXECUTION_POLICY_RECORD_INVALID")
    return {key: value for key, value in expected.items() if key != "policy_source"}


def guard_v2(requested, *, protocol, explicit):
    result = identity(requested, algorithm="wgm", profile="v2", protocol=protocol, explicit=explicit)
    if requested == "origin":
        warnings.warn("WGM V2 is running with origin_hgm_v1 failure semantics.\n"
                      "Anytime confidence guarantees do not apply.", stacklevel=2)
    return result


def check_resume(directory, config):
    """Run before preflight/callbacks; never add metadata to a historical run."""
    path = Path(directory) / "run_manifest.json"
    if not path.is_file():
        raise ExecutionPolicyError("EXECUTION_POLICY_LEGACY_RESUME_UNAVAILABLE")
    saved = json.loads(path.read_text()).get("execution_policy")
    if saved is None:
        if validated(config):
            raise ExecutionPolicyError("EXECUTION_POLICY_LEGACY_RESUME_UNAVAILABLE")
        return  # The ordinary origin reader retains its existing compatibility checks.
    if semantics(saved) != semantics(record(config)):
        raise ExecutionPolicyError("EXECUTION_POLICY_RESUME_MISMATCH")
