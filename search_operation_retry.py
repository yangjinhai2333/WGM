"""Operator-authorized retry of a failed, side-effect-free diagnosis attempt.

A search operation whose latest diagnosis call ended unsuccessfully is blocked
(BLOCK_INFRASTRUCTURE_FAILURE) and never retried automatically. Diagnosis runs
read-only (read-only sandbox, Read/Glob/Grep tools only), so a failed attempt
cannot have changed any workspace. After review, this explicit command appends
a `diagnosis_abandoned` event with an audit artifact: the attempt then counts as
an invalid diagnosis attempt, and the operation resumes under the existing retry
budget (the original HGM rule counts a failed call as a used attempt). The
failed attempt's events and call record are kept unchanged.
"""
from datetime import datetime, timezone
import json
from pathlib import Path

import evaluation_ledger as io
from evaluation_plan import check
from run_metadata import file_hash


def _blocked_operation(journal):
    from search_operation_journal import classify
    blocked = []
    for plan in journal.plans():
        state, _ = journal.inspect(plan)
        if classify(state) == "BLOCK_INFRASTRUCTURE_FAILURE":
            blocked.append((plan, state))
    check(len(blocked) == 1, "RETRY_REQUIRES_ONE_BLOCKED_OPERATION")
    return blocked[0]


def abandon_failed_diagnosis(journal, plan, state, body):
    """Journal part: audit artifact plus the diagnosis_abandoned event (caller holds the run lease)."""
    attempt = state["diagnosis"][-1]["attempt"]
    with journal.session(plan) as session:
        check(session.state == state, "RETRY_STATE_CHANGED")
        artifact = session.artifact(f"diagnosis-abandoned-{attempt:06d}.json",
                                    {"kind": "diagnosis_abandoned", "attempt": attempt, "authorization": body})
        session.append("diagnosis_abandoned", {"attempt": attempt, "artifact": artifact})
    return artifact


def authorize_diagnosis_retry(directory, *, acknowledge=False):
    from evaluation_scheduler import ExecutionRun
    from search_operation_journal import SearchOperationJournal, successful
    check(acknowledge, "RETRY_ACKNOWLEDGEMENT_REQUIRED")
    directory = io._checked_path(directory)
    manifest = io._read_json(directory / "run_manifest.json")
    check(manifest["codex"].get("provider") == "claude_code", "RETRY_PROVIDER_UNSUPPORTED")
    run = ExecutionRun(directory, manifest)
    with run.lock():
        journal = SearchOperationJournal(directory, manifest["run_id"], manifest["reproducibility_fingerprint"])
        plan, state = _blocked_operation(journal)
        check(state["blocked"] is None and not state["generation"] and state["diagnosis"], "RETRY_NOT_A_DIAGNOSIS_FAILURE")
        attempt = state["diagnosis"][-1]
        check("marker" in attempt and "terminal" in attempt and "result" not in attempt
              and not successful(attempt["terminal"]["evidence"]), "RETRY_NOT_A_DIAGNOSIS_FAILURE")
        evidence = attempt["terminal"]["evidence"]
        check(evidence["started"] and evidence["reaped"] and not evidence["timed_out"] and not evidence["interrupted"],
              "RETRY_TERMINAL_NOT_TRUSTED")
        call_id = attempt["marker"]["call_id"]
        record_path = directory / "codex_calls" / f"{call_id}.json"
        record = io._read_json(record_path)
        check(record.get("phase") == "diagnosis" and record.get("provider") == "claude_code"
              and record.get("status") == "failed" and record.get("successful_calls") == 0,
              "RETRY_CALL_RECORD_MISMATCH")
        check(all(r.get("tools_used") == [] and r.get("tool_calls") == 0 for r in record.get("rate_limit_rejections", [])),
              "RETRY_SIDE_EFFECTS_POSSIBLE")
        body = {"call_id": call_id,
                "call_record_sha256": file_hash(record_path), "terminal_evidence": evidence,
                "reason": "failed_read_only_diagnosis_no_side_effects",
                "diagnosis_is_read_only": True, "authorized_at_utc": datetime.now(timezone.utc).isoformat()}
        artifact = abandon_failed_diagnosis(journal, plan, state, body)
        return {"operation_id": plan.operation_id, "attempt": attempt["attempt"], "call_id": call_id,
                "artifact": artifact, "remaining_diagnosis_attempts":
                plan.identity["diagnosis"]["max_attempts"] - len(state["diagnosis"])}
