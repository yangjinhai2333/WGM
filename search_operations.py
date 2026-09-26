"""Validated diagnosis/generation lifecycle; origin never dispatches here."""
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from subprocess import TimeoutExpired

import numpy as np

import evaluation_ledger as io
from candidate_workspace import CandidateError
from evaluation_plan import durable_json
from network_readiness import ReadinessAbort, await_runner
from runners import CodexExecConfig, CodexExecRequest, CodexExecRunner
from runners.codex_exec import CodexRunnerError
from runners.usage import invoke
from search_config import is_corrected
from search_state import rng_state
from search_operation_journal import (SearchOperationJournal, SearchOperationPlan, OperationError,
                                      call_id, classify, require, successful, validate_rng, _terminal)
from search_operation_workspace import OperationWorkspace, sha, verify_source_links


def operation_rng(state):
    if state.config.search.algorithm == "wgm_v2":
        value = state.rng_envelope()
        validate_rng(value)
        return value
    value = {"schema_version": 1, "global": rng_state(), "selector": None if state.selector is None else
            {"mode": "global_numpy" if state.selector.rng is np.random else "private_numpy",
             "state": state.selector.state_dict()["rng"]}}
    value = io._decode(io.canonical(value))
    validate_rng(value)
    return value


def checkpoint_rng(state):
    if state.get("v2_schema") == 1:
        return deepcopy(state["v2_rng"])
    return deepcopy({"schema_version": 1, "global": state["rng"], "selector": None if state["selector"] is None else
            {"mode": state["execution"].get("selector_rng_mode", "global_numpy"),
             "state": state["selector"]["rng"]}})


def read_checkpoint(directory):
    row = io._read_json(Path(directory) / "search_state.json")
    io._keys(row, ("state", "sha256", "fingerprint", "runtime_control") if row.get("state", {}).get("v2_schema") == 1
             else ("state", "sha256", "fingerprint"))
    require(row["sha256"] == io.digest(row["state"]), "BLOCK_IDENTITY_MISMATCH")
    execution = row["state"].get("execution")
    require(type(execution) is dict and type(execution.get("search_operation_schema")) is int
            and execution["search_operation_schema"] == 1, "BLOCK_CORRUPT_OPERATION")
    validate_rng(checkpoint_rng(row["state"]))
    pending = row["state"]["inflight"]
    if pending and pending["kind"] == "expansion" and "operation_plan" in pending:
        plan = SearchOperationPlan.from_dict(pending["operation_plan"])
        require(io.digest(checkpoint_rng(row["state"])) == plan.identity["post_rng_digest"],
                "BLOCK_IDENTITY_MISMATCH")
    return row


def diagnosis_prompt(config, sources, directory, entries, entry, parent, *, legacy_builder=None):
    from provider_prompts import context_system, provider_of, runtime_boundary
    from candidate_provenance import read_chain
    chain = read_chain(directory, parent, require_versioned=True)
    patches = [str(Path(directory) / item.provenance.candidate_id / "model_patch.diff") for item in chain]
    dataset = list(entries.values())
    root = str(sources.root)
    if is_corrected(config):
        from diagnosis_context import build_parts, bounded_prompt
        parts = build_parts(entry, parent, root, str(directory), dataset, patches,
                            polyglot=config.evaluation.polyglot)
        return bounded_prompt(parts, config.diagnosis_context.max_input_chars,
                              system=context_system(provider_of(config.codex)))[0]
    from prompts.self_improvement_prompt import get_diagnose_prompt_polyglot, get_diagnose_prompt_swe
    if config.evaluation.polyglot:
        # Every algorithm shares the original Polyglot diagnosis prompt (no DGM-specific variant exists).
        builder = get_diagnose_prompt_polyglot
    else:
        builder = legacy_builder or get_diagnose_prompt_swe
    from downstream_solver import load_agent_policy
    if config.evaluation.polyglot:
        # The original Polyglot prompt draws random.random() (25% stochasticity
        # variant). A validated operation must rebuild the identical prompt and
        # never advance the search RNG, so the draw uses a private generator
        # seeded by the operation's inputs; the global RNG is left untouched.
        import hashlib as _hashlib
        import random as _random
        seed = int(_hashlib.sha256(f"{Path(directory).name}\0{parent}\0{entry}".encode()).hexdigest()[:16], 16)
        private, original = _random.Random(seed), _random.random
        _random.random = private.random
        try:
            system, user = builder(entry, parent, root, str(directory), dataset, patch_files=patches)
        finally:
            _random.random = original
    else:
        system, user = builder(entry, parent, root, str(directory), dataset, patch_files=patches)
    policy = load_agent_policy(root, patches, polyglot=config.evaluation.polyglot)
    system += (
        runtime_boundary(provider_of(config.codex)) +
        f"<current_downstream_instructions>\n{policy}\n</current_downstream_instructions>"
    )
    return (f"<system_instructions>\n{system}\n</system_instructions>\n\n"
            f"<diagnosis_request>\n{user}\n</diagnosis_request>\n\n"
            "<response_format>\nReturn only a JSON object matching the supplied schema: "
            "exactly implementation_suggestion and problem_description, both nonempty strings. "
            "Preserve the proposed improvement in these fields. Do not emit Markdown fences "
            "or the other intermediate analysis fields requested above.\n</response_format>")


def diagnosis_result(raw):
    """Only bounded, schema-checked fields may cross from quarantine to artifacts."""
    from self_improve_step import DiagnosisValidationError, validate_diagnosis
    from diagnosis_context import redact_log
    # Missing/unreadable/unbounded local output is not model schema evidence.
    require(type(raw) is bytes and len(raw) <= 256 * 1024, "DIAGNOSIS_STAGING_INVALID")
    invalid = {"kind": "diagnosis_result", "outcome": "invalid", "fields": None, "result_digest": io.digest(None)}
    def pairs(rows):
        value = {}
        for key, item in rows:
            if key in value:
                raise DiagnosisValidationError("duplicate field")
            value[key] = item
        return value
    def constant(_):
        raise DiagnosisValidationError("non-JSON constant")
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)
    except (json.JSONDecodeError, UnicodeDecodeError, DiagnosisValidationError):
        return invalid
    try:
        validate_diagnosis(value)
    except DiagnosisValidationError:
        return invalid
    if not all(len(text) <= 65536 and redact_log(text) == "\n".join(text.splitlines())
               for text in value.values()):
        return invalid
    return {"kind": "diagnosis_result", "outcome": "valid", "fields": value, "result_digest": io.digest(value)}


class SearchOperations:
    def __init__(self, run, config, sources, entries, recorder=None, *, diagnosis_runner=None,
                 generation_runner=None, workspace=None, choose_entry=None, prompt_builder=None,
                 fault=lambda point: None):
        self.run, self.config, self.sources, self.entries, self.recorder = run, config, sources, entries, recorder
        self.fault = fault
        self.journal = SearchOperationJournal(run.directory, run.manifest["run_id"],
            run.manifest["reproducibility_fingerprint"], fault=fault)
        require(config.execution.policy == "validated", "BLOCK_IDENTITY_MISMATCH")
        require(config.llm.diagnose_backend == config.llm.self_improve_backend == "codex_exec",
                "BLOCK_IDENTITY_MISMATCH")
        require(not any(arg.split("=")[0] == "--add-dir" for arg in config.codex.extra_args),
                "BLOCK_IDENTITY_MISMATCH")
        from runners.codex_exec import _validate_timeout
        candidate = config.candidate
        require(type(candidate.max_patch_bytes) is int and candidate.max_patch_bytes > 0
                and type(candidate.keep_failed_workspace) is bool
                and isinstance(candidate.validation_command, list)
                and all(isinstance(arg, str) and bool(arg) for arg in candidate.validation_command),
                "BLOCK_IDENTITY_MISMATCH")
        _validate_timeout(candidate.validation_timeout_seconds)
        self.diagnosis_runner = diagnosis_runner or CodexExecRunner(replace(config.codex,
            sandbox="read-only", approve_for_me=False, ephemeral=True, output_schema=None))
        self.generation_runner = generation_runner or CodexExecRunner(replace(config.codex,
            sandbox="workspace-write", approve_for_me=True, ephemeral=True, output_schema=None))
        from provider_prompts import provider_of
        self.workspace = workspace or OperationWorkspace(run.directory, sources, config.candidate, fault=fault,
                                                         prompt_provider=provider_of(config.codex))
        self.choose_entry = choose_entry
        self.prompt_builder = prompt_builder or diagnosis_prompt

    def verify_runners(self):
        for stage, runner in (("diagnosis", self.diagnosis_runner), ("generation", self.generation_runner)):
            expected = replace(self.config.codex, sandbox="read-only" if stage == "diagnosis" else "workspace-write",
                               approve_for_me=stage == "generation", ephemeral=True, output_schema=None)
            require(isinstance(getattr(runner, "config", None), CodexExecConfig), "BLOCK_IDENTITY_MISMATCH")
            actual, frozen = asdict(runner.config), asdict(expected)
            actual.pop("executable")
            frozen.pop("executable")
            require(io.same_json(actual, frozen), "BLOCK_IDENTITY_MISMATCH")

    def attach(self, engine):
        execution = engine.state.execution
        require("search_operation_schema" in execution or not engine.state.history, "BLOCK_CORRUPT_OPERATION")
        execution.setdefault("search_operation_schema", 1)
        execution.setdefault("consumed_operations", {})
        mode = ("v2_domains" if self.config.search.algorithm == "wgm_v2" else
                "private_numpy" if engine.state.selector and engine.state.selector.rng is not np.random else "global_numpy")
        execution.setdefault("selector_rng_mode", mode)
        require(execution["search_operation_schema"] == 1 and execution["selector_rng_mode"] == mode)

    def fingerprints(self, parent):
        from self_improve_step import DIAGNOSIS_SCHEMA
        from run_metadata import effective_configuration, fingerprint_inputs
        root = Path(__file__).parent
        builders = ("search_operations.py", "search_operation_workspace.py", "self_improve_step.py",
                    "diagnosis_context.py", "prompts/self_improvement_prompt.py", "prompts/testrepo_prompt.py",
                    "search_operation_journal.py", "candidate_workspace.py", "candidate_provenance.py",
                    "validated_search.py", "search_state.py", "tree.py", "wgm_selector.py", "task_schedule.py",
                    "runners/codex_exec.py", "runners/usage.py")
        if self.config.search.algorithm == "wgm_v2":
            builders += ("wgm_v2_config.py", "wgm_v2_selector.py", "wgm_v2_state.py", "wgm_v2_search.py")
        if self.config.search.algorithm == "dgm":
            builders += ("dgm_legacy_config.py", "dgm_legacy_selector.py", "dgm_legacy_operations.py", "dgm_legacy_search.py", "dgm_legacy_prompts.py")
        proposed = deepcopy(self.run.manifest)
        proposed["configuration"]["effective"] = effective_configuration(self.config, self.config.search.algorithm)
        model = asdict(self.config.codex)
        model.pop("executable")  # Location is runtime metadata, as in the run fingerprint.
        hashes = {name: sha((root / name).read_bytes()) for name in builders}
        return {"configuration": io.digest(fingerprint_inputs(proposed)["configuration"]),
            "model": io.digest(model), "policy": parent["policy_fingerprint"],
            "source": parent["source_fingerprint"],
            "builder": io.digest(hashes),
            "schema": io.digest(DIAGNOSIS_SCHEMA), "task_schedule": io.digest(self.run.manifest["tasks"])}

    def prompt(self, entry, parent):
        return self.prompt_builder(self.config, self.sources, self.run.directory, self.entries, entry, parent)

    def task(self, entry, parent):
        actual = entry
        if entry.startswith("solve_") and is_corrected(self.config):
            own = io._read_json(self.run.directory / parent / "metadata.json")["overall_performance"]
            choices = own["total_unresolved_ids"] or own["total_resolved_ids"]
            actual = sorted(choices)[0] if choices else entry
        return actual, self.entries.get(actual, {})

    def plan(self, engine, parent, cid, checkpoint):
        if self.choose_entry is None:
            from hgm_utils import choose_entry
            entry = choose_entry(parent.commit_id)
        else:
            entry = self.choose_entry(parent.commit_id)
        source = self.sources.get(parent.commit_id)
        verify_source_links(source["path"])
        task, row = self.task(entry, parent.commit_id)
        prompt = self.prompt(entry, parent.commit_id)
        fingerprints = self.fingerprints(source)
        pre, post = checkpoint_rng(checkpoint["state"]), operation_rng(engine.state)
        return SearchOperationPlan.create(run_id=self.journal.run_id, ordinal=len(engine.state.history)-1,
            algorithm=self.config.search.algorithm, profile=("wgm_exp3_bai_v2" if self.config.search.algorithm == "wgm_v2"
                else "paper_corrected" if is_corrected(self.config) else "legacy_code"),
            policy="validated_observation_v2", run_fingerprint=self.journal.fingerprint,
            parent={"node_id": parent.id, "candidate_id": parent.commit_id,
                    "branch_id": engine.state.branches[parent.id], "provenance_digest": source["chain_digest"],
                    "source_digest": source["source_fingerprint"]}, candidate_id=cid,
            diagnosis={"entry": entry, "task_id": task, "entry_digest": io.digest(row), "commit": parent.commit_id,
                       "base_commit": row.get("base_commit"), "prompt_digest": sha(prompt.encode()), "max_attempts": 3},
            generation={"prompt_recipe_digest": io.digest({"builder": fingerprints["builder"], "schema": 1,
                "test_description": "swe_self_improvement", "candidate_id": cid})}, fingerprints=fingerprints,
            budget={"used": engine.count, "limit": self.config.execution.max_task_evals,
                    "initial_attempts": engine.state.initial_attempts, "initialized": engine.state.initialized,
                    "expansion": deepcopy(engine.state.history[-1])},
            pre_rng=pre, post_rng=post, pre_rng_digest=io.digest(pre), post_rng_digest=io.digest(post),
            checkpoint_digest=io.digest({k: v for k, v in checkpoint.items() if k != "runtime_control"}))

    def verify(self, plan, engine, session=None, *, committed=False):
        self.journal.validate(plan)
        self.verify_runners()
        body = plan.identity
        source = self.sources.get(body["parent"]["candidate_id"])
        verify_source_links(source["path"])
        require(self.fingerprints(source) == body["fingerprints"]
                and source["chain_digest"] == body["parent"]["provenance_digest"], "BLOCK_IDENTITY_MISMATCH")
        # Never repeat task selection. Parent evaluation views can grow after a
        # committed operation; the frozen dataset row remains the identity.
        row = self.entries.get(body["diagnosis"]["task_id"], {})
        require(io.digest(row) == body["diagnosis"]["entry_digest"]
                and row.get("base_commit") == body["diagnosis"]["base_commit"],
                "BLOCK_IDENTITY_MISMATCH")
        if not committed:
            saved = read_checkpoint(self.run.directory)
            pending = saved["state"]["inflight"]
            require(saved["fingerprint"] == self.journal.fingerprint and pending
                    and pending.get("operation_plan") == plan.to_dict(), "BLOCK_IDENTITY_MISMATCH")
            require(io.digest(checkpoint_rng(saved["state"])) == body["post_rng_digest"]
                    and io.digest(operation_rng(engine.state)) == body["post_rng_digest"], "BLOCK_IDENTITY_MISMATCH")
            require(engine.state.inflight == pending and engine.count == body["budget"]["used"]
                    and engine.state.history[-1] == body["budget"]["expansion"], "BLOCK_IDENTITY_MISMATCH")
            live = io._decode(io.canonical(engine.state.state_dict(engine.count)))
            require(io.same_json(live, saved["state"]), "BLOCK_IDENTITY_MISMATCH")
            nodes = engine.state.nodes
            require(list(nodes) == list(range(len(nodes))), "BLOCK_IDENTITY_MISMATCH")
            for node_id, node in nodes.items():
                require(node.id == node_id and [child.id for child in node.children] ==
                        [n.id for n in nodes.values() if n.parent_id == node_id], "BLOCK_IDENTITY_MISMATCH")
            parent = engine.state.nodes.get(body["parent"]["node_id"])
            require(parent is not None and parent.commit_id == body["parent"]["candidate_id"]
                    and engine.state.branches[parent.id] == body["parent"]["branch_id"], "BLOCK_IDENTITY_MISMATCH")
        if session is not None:
            session.refresh()

    def _invoke(self, session, engine, stage, prompt):
        runner = self.diagnosis_runner if stage == "diagnosis" else self.generation_runner
        attempt = session.state[stage][-1]["attempt"]
        require("marker" not in session.state[stage][-1])
        try:
            receipt = await_runner(runner, recorder=self.recorder)
        except ReadinessAbort:
            session.append("infrastructure_failed", {"code": "READINESS_ABORTED"})
            raise
        self.fault(stage + ".after_readiness")
        self.verify(session.plan, engine, session)
        require(sha(prompt.encode()) == session.state[stage][-1]["prompt_digest"], "BLOCK_IDENTITY_MISMATCH")
        if stage == "generation":
            self.workspace.verify(session, pristine=True)
        try:
            # Rechecks authority/TTL/cancellation only, never a second probe.
            await_runner(runner, receipt, reject_stale=True)
        except ReadinessAbort:
            session.append("infrastructure_failed", {"code": "READINESS_ABORTED"})
            raise
        cid = call_id(session.plan, stage, attempt)
        session.append(stage + "_marker", {"attempt": attempt, "call_id": cid,
            "receipt": None if receipt is None else asdict(receipt)})
        self.fault(stage + ".after_marker")
        owner = session.diagnosis_staging(attempt) if stage == "diagnosis" else None

        def started():
            session.append(stage + "_started", {"attempt": attempt})
            self.fault(stage + ".after_started")

        def terminal(proof):
            evidence = asdict(proof)
            _terminal(evidence)
            require(evidence["started"] == session.state[stage][-1].get("started", False),
                    "BLOCK_AMBIGUOUS_" + stage.upper())
            if not successful(evidence):
                result = {"kind": "process_failure"}
            elif stage == "diagnosis":
                result = diagnosis_result(owner.read())
            else:
                result = self.workspace.snapshot(session)
            self.fault(stage + ".before_result_artifact")
            artifact = session.artifact(f"{stage}-attempt-{attempt:06d}.json", result)
            self.fault(stage + ".after_result_artifact")
            if owner is not None:
                owner.cleanup()
            self.fault(stage + ".before_terminal")
            session.append(stage + "_terminal", {"attempt": attempt, "evidence": evidence, "artifact": artifact})
            self.fault(stage + ".after_terminal")

        from self_improve_step import DIAGNOSIS_SCHEMA
        request = CodexExecRequest(prompt, self.sources.root if stage == "diagnosis" else self.workspace.paths(session)[1],
            on_started=started, on_terminal=terminal, call_id=cid, readiness_receipt=receipt,
            last_message_path=owner.path if owner is not None else None, last_message_owner=owner)
        try:
            invoke(runner, request, "diagnosis" if stage == "diagnosis" else "self_improvement", recorder=self.recorder,
                   schema=deepcopy(DIAGNOSIS_SCHEMA) if stage == "diagnosis" else None)
        except (CodexRunnerError, OSError):
            # A trusted terminal remains authoritative even if JSON parsing or
            # accounting failed later. A marker without one must never retry.
            if session.poisoned:
                raise
        session.refresh()
        require("terminal" in session.state[stage][-1], "BLOCK_AMBIGUOUS_" + stage.upper())
        require(successful(session.state[stage][-1]["terminal"]["evidence"]), "BLOCK_INFRASTRUCTURE_FAILURE")

    def publish_plan(self, plan, engine):
        self.journal.publish(plan)

    def execute(self, plan, engine):
        self.publish_plan(plan, engine)
        with self.journal.session(plan) as session:
            self.verify(plan, engine, session)
            while True:
                decision = classify(session.state)
                if decision.startswith("BLOCK_"):
                    raise OperationError(decision)
                if session.state["terminal"]:
                    self.workspace.publish(session, session.state["terminal"])
                    return session.state["terminal"]
                attempts = session.state["diagnosis"]
                current = attempts[-1] if attempts else None
                if current and "terminal" in current and "result" not in current:
                    artifact = session.journal.read_artifact(plan, current["terminal"]["artifact"])
                    require(artifact["kind"] == "diagnosis_result")
                    require(artifact["result_digest"] == io.digest(artifact["fields"]))
                    session.append("diagnosis_result", {"attempt": current["attempt"],
                        "outcome": artifact["outcome"], "artifact": current["terminal"]["artifact"]})
                    self.fault("diagnosis.after_seal")
                    continue
                if not current or current.get("result", {}).get("outcome") == "invalid":
                    if len(attempts) == plan.identity["diagnosis"]["max_attempts"]:
                        artifact = session.artifact("diagnosis-invalid.json", {"kind": "diagnosis_invalid"})
                        session.append("operation_terminal", {"outcome": "diagnosis_invalid", "artifact": artifact,
                            "candidate_digest": artifact["sha256"]})
                        continue
                    session.append("diagnosis_reserved", {"attempt": len(attempts) + 1,
                        "prompt_digest": plan.identity["diagnosis"]["prompt_digest"]})
                    self.fault("diagnosis.after_reserved")
                    current = session.state["diagnosis"][-1]
                if "result" not in current:
                    prompt = self.prompt(plan.identity["diagnosis"]["entry"], plan.identity["parent"]["candidate_id"])
                    self._invoke(session, engine, "diagnosis", prompt)
                    continue
                from prompts.self_improvement_prompt import get_problem_description_prompt
                artifact = session.journal.read_artifact(plan, current["result"]["artifact"])
                problem = get_problem_description_prompt(artifact["fields"], False)
                prompt = self.workspace.prompt(session, problem)
                if not session.state["generation"]:
                    session.append("generation_reserved", {"attempt": 1, "prompt_digest": sha(prompt.encode())})
                    self.fault("generation.after_reserved")
                generation = session.state["generation"][-1]
                require(generation["prompt_digest"] == sha(prompt.encode()), "BLOCK_IDENTITY_MISMATCH")
                if "terminal" not in generation:
                    try:
                        self.workspace.prepare(session)
                    except (OSError, CandidateError, TimeoutExpired):
                        session.append("infrastructure_failed", {"code": "PREPARATION_FAILED"})
                        raise OperationError("BLOCK_INFRASTRUCTURE_FAILURE") from None
                    self._invoke(session, engine, "generation", prompt)
                    continue
                try:
                    result = self.workspace.seal(session, generation["terminal"]["artifact"])
                except (OSError, TimeoutExpired):
                    session.append("infrastructure_failed", {"code": "VALIDATION_IO"})
                    raise OperationError("BLOCK_INFRASTRUCTURE_FAILURE") from None
                session.append("operation_terminal", result)
                self.fault("operation.after_terminal")

    def commit(self, engine, plan, terminal):
        """Exactly one SearchState checkpoint consumes admission/invalid feedback."""
        consumed = engine.state.execution["consumed_operations"]
        binding = {"operation_terminal_digest": terminal["terminal_digest"],
                   "candidate_digest": terminal["candidate_digest"]}
        if plan.operation_id in consumed:
            require(consumed[plan.operation_id] == binding, "BLOCK_IDENTITY_MISMATCH")
            with self.journal.session(plan) as session:
                self.verify(plan, engine, session, committed=True)
                require(session.state["terminal"] == terminal, "BLOCK_IDENTITY_MISMATCH")
                self.workspace.publish(session, terminal, committed=True)
            return
        with self.journal.session(plan) as session:
            self.verify(plan, engine, session)
            require(session.state["terminal"] == terminal, "BLOCK_IDENTITY_MISMATCH")
            self.workspace.publish(session, terminal)
            self.fault("operation.before_commit")
            try:
                child = None
                if terminal["outcome"] == "candidate":
                    cid, pid = plan.identity["candidate_id"], plan.identity["parent"]["node_id"]
                    require(not any(n.commit_id == cid for n in engine.state.nodes.values()), "BLOCK_IDENTITY_MISMATCH")
                    self.fault("admission.before_node")
                    child = engine.node_type(cid, parent_id=pid)
                    self.fault("admission.after_node")
                    self.fault("admission.before_parent")
                    engine.state.nodes[pid].add_child(child)
                    self.fault("admission.after_parent")
                self.fault("admission.before_bookkeeping")
                engine.state.end_expansion(child)
                self.fault("admission.after_bookkeeping")
                consumed[plan.operation_id] = binding
                self.fault("admission.after_consumption")
                payload = io._decode(io.canonical(engine.state.state_dict(engine.count)))
                durable_json(self.run.directory / "search_state.json",
                    {"state": payload, "sha256": io.digest(payload), "fingerprint": engine.fingerprint,
                     **({"runtime_control": deepcopy(engine.state.runtime_control)}
                        if self.config.search.algorithm == "wgm_v2" else {})},
                    replace_existing=True, fault=lambda point: self.fault("operation.commit." + point))
            except BaseException:
                # Includes constructor/admission errors: never turn these into
                # candidate-invalid, nor leave half-mutated in-memory rewards.
                engine.restore()
                raise
            self.fault("operation.after_commit")

    def reconcile(self, engine):
        pending = engine.state.inflight
        if pending and pending["kind"] == "expansion":
            require("operation_plan" in pending, "BLOCK_CORRUPT_OPERATION")
            plan = SearchOperationPlan.from_dict(pending["operation_plan"])
            self.journal.publish(plan)
        consumed = engine.state.execution["consumed_operations"]
        plans = self.journal.plans()
        require(set(consumed).issubset(p.operation_id for p in plans), "BLOCK_CORRUPT_OPERATION")
        for plan in plans:
            state, _ = self.journal.inspect(plan)
            if plan.operation_id in consumed:
                self.verify(plan, engine, committed=True)
                terminal = state["terminal"]
                require(terminal is not None and consumed[plan.operation_id] == {
                    "operation_terminal_digest": terminal["terminal_digest"],
                    "candidate_digest": terminal["candidate_digest"]}, "BLOCK_IDENTITY_MISMATCH")
                with self.journal.session(plan) as session:
                    self.workspace.publish(session, terminal, committed=True)
            else:
                require(pending is not None and pending.get("operation_plan") == plan.to_dict(),
                        "BLOCK_CORRUPT_OPERATION")
                result = self.execute(plan, engine)
                self.commit(engine, plan, result)
