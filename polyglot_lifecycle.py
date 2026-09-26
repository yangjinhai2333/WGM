"""Validated (ledger) evaluation of Polyglot tasks.

The original Polyglot harness (polyglot/harness.py) prepares the language
image, lets the agent edit /testbed, then applies the hidden test patch and
passes a task when the language's TEST_COMMANDS exit with status 0. This module
keeps exactly that semantics and adds what the validated lifecycle adapter
needs (downstream_lifecycle): preparation/image fingerprints, restore and
verified cleanup, and a grader for container_evaluator_result.

It is a separate module so that the evaluator identity files of the SWE
condition (downstream_solver.py, swe_bench/harness.py, polyglot/harness.py) are
unchanged. The Polyglot evaluator fingerprint covers this module and the
Polyglot harness/build/spec files it relies on.
"""
import datetime
import hashlib
from pathlib import Path
import re
import threading
from types import SimpleNamespace

import docker

from downstream_solver import TaskFailure, container_command, image_facts, load_agent_policy, put_bytes, run_task
from polyglot.constants import TEST_COMMANDS
from polyglot.harness import get_eval_script
from prompts.testrepo_prompt import get_test_description
from swe_bench.utils import close_logger, setup_logger

ROOT = Path(__file__).resolve().parent
EVALUATOR_FILES = ("polyglot_lifecycle.py", "polyglot/harness.py", "polyglot/constants.py",
                   "polyglot/test_spec.py", "polyglot/docker_build.py", "polyglot/dockerfiles.py")
TEST_PATCH_PATH = "/tmp/wgm-tests.patch"
INNER_SCRIPT_PATH = "/tmp/wgm-polyglot-eval.sh"
_image_build_lock = threading.Lock()


def evaluator_fingerprint(root=ROOT):
    digest = hashlib.sha256()
    for name in EVALUATOR_FILES:
        digest.update(name.encode() + b"\0" + (Path(root) / name).read_bytes() + b"\0")
    return digest.hexdigest()


def wrapper_script():
    """Apply the hidden tests, then run the original script; report both outcomes."""
    return "\n".join([
        "#!/bin/bash",
        "cd /testbed",
        f"if [ -s {TEST_PATCH_PATH} ]; then",
        f"  if ! git apply --check {TEST_PATCH_PATH} || ! git apply {TEST_PATCH_PATH}; then",
        '    echo "WGM_POLYGLOT_TESTPATCH=failed"; exit 97',
        "  fi",
        "fi",
        'echo "WGM_POLYGLOT_TESTPATCH=applied"',
        f"/bin/bash {INNER_SCRIPT_PATH}",
        "rc=$?",
        'echo "WGM_POLYGLOT_EXIT=$rc"',
        "exit $rc",
    ]) + "\n"


def grade(testspec, prediction, path, include_tests_status=True):
    """Pass iff the hidden tests applied and the language test commands exited 0."""
    text = Path(path).read_text(errors="replace")
    applied = re.search(r"^WGM_POLYGLOT_TESTPATCH=applied$", text, re.M) is not None
    codes = re.findall(r"^WGM_POLYGLOT_EXIT=(\d+)$", text, re.M)
    resolved = applied and bool(codes) and codes[-1] == "0"
    # The model patch itself was applied by the lifecycle before evaluation;
    # a hidden test patch that no longer applies is an unresolved task (as in SWE).
    return {prediction["instance_id"]: {"resolved": resolved, "patch_successfully_applied": True,
                                        "tests_applied": applied, "exit_code": int(codes[-1]) if codes else None}}


def _codex_callbacks(entry, evaluation_timeout, *, evaluation_id=None):
    state = {}
    inner = get_eval_script(TEST_COMMANDS[entry["language"]])

    def fingerprints(client, spec):
        from evaluation_ledger import digest
        prepare.testspec_fingerprint = digest({
            "task_id": entry["instance_id"], "script": inner, "wrapper": wrapper_script(),
            "image": spec.instance_image_key,
            "test_patch_sha256": hashlib.sha256(entry["test_patch"].encode()).hexdigest()})
        facts = image_facts(client, spec)
        prepare.image_fingerprint = digest(facts)
        return facts

    def cleanup(container):
        try:
            try:
                container.stop(timeout=10)
            finally:
                container.remove(force=True)
        finally:
            if "client" in state:
                state["client"].close()
            if "logger" in state:
                close_logger(state["logger"])
        prepare.cleanup_verified = True

    def prepare(log_dir):
        prepare.cleanup_verified = False
        from polyglot.docker_build import build_container, build_env_images
        from polyglot.test_spec import make_test_spec
        client = state["client"] = docker.from_env(timeout=evaluation_timeout + 60)
        container = None
        try:
            spec = state["spec"] = make_test_spec(entry)
            logger = state["logger"] = setup_logger(str(Path(log_dir) / "docker.log"))
            try:
                client.images.get(spec.instance_image_key)
                cached = True
            except docker.errors.ImageNotFound:
                cached = False
            with _image_build_lock:
                _, failed = build_env_images(client, dataset=[entry], force_rebuild=False, max_workers=1)
                if failed:
                    raise TaskFailure("Polyglot environment image preparation failed")
                container = build_container(spec, client, evaluation_id or datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
                                            logger, False, force_rebuild=False)
            container.start()
            facts = fingerprints(client, spec)
            return container, get_test_description(eval_script=inner, polyglot=True), cached, facts
        except BaseException:
            if container is not None:
                cleanup(container)
            else:
                client.close()
                if "logger" in state:
                    close_logger(state["logger"])
                prepare.cleanup_verified = True
            raise

    def restore(resource_id, log_dir, *, cleanup=False):
        # Only look up this attempt's owned resource; never create/start one.
        client = state.get("client")
        if client is None:
            client = state["client"] = docker.from_env(timeout=evaluation_timeout + 60)
        container = client.containers.get(resource_id)
        if cleanup:
            return container, ""
        from polyglot.test_spec import make_test_spec
        spec = state.get("spec")
        if spec is None:
            spec = state["spec"] = make_test_spec(entry)
        fingerprints(client, spec)
        return container, get_test_description(eval_script=inner, polyglot=True)

    def evaluate(container, prediction, log_dir):
        raise TaskFailure("Polyglot validated evaluation uses the lifecycle evaluator only")

    def lifecycle_evaluate(container, prediction, log_dir, *, on_started):
        from downstream_lifecycle import container_evaluator_result
        # Files only; nothing runs before container_evaluator_result confirms the start.
        put_bytes(container, TEST_PATCH_PATH, entry["test_patch"].encode())
        put_bytes(container, INNER_SCRIPT_PATH, inner.encode())
        return container_evaluator_result(container, prediction, SimpleNamespace(eval_script=wrapper_script()),
                                          evaluation_timeout, grade, on_started, log_root=log_dir)

    prepare.restore = restore
    evaluate.lifecycle = lifecycle_evaluate
    evaluate.evaluator_fingerprint = evaluator_fingerprint()
    return prepare, evaluate, cleanup


def process_entry(entry, out_dname, model_name_or_path, model_patch_paths=None,
                  init_agent_path=".", *, backend="codex_exec", codex_config=None,
                  evaluation_timeout=None, runner=None, policy=None, recorder=None,
                  execution_spec=None, evaluation_ledger=None, source_fingerprint=None):
    """Validated counterpart of polyglot.harness.process_entry (ledger path only)."""
    if backend != "codex_exec" or execution_spec is None or evaluation_ledger is None:
        raise ValueError("Polyglot lifecycle evaluation requires the validated codex_exec path")
    if policy is None or source_fingerprint is None:
        raise ValueError("Evaluation lifecycle requires verified policy and source identity")
    limit = evaluation_timeout if evaluation_timeout is not None else 1800
    prepare, evaluate, cleanup = _codex_callbacks(entry, limit, evaluation_id=execution_spec.identity.evaluation_id)
    return run_task(entry, out_dname, model_name_or_path, prepare=prepare, evaluate=evaluate, cleanup=cleanup,
                    runner=runner, codex_config=codex_config, timeout=limit, policy=policy, recorder=recorder,
                    execution_spec=execution_spec, evaluation_ledger=evaluation_ledger,
                    source_fingerprint=source_fingerprint)
