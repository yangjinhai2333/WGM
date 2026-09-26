import argparse
import datetime
import json
import os
import random
import re
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import stdev

from candidate_workspace import generate_candidate
from config import HGMConfig, validate_self_improve_backend

import docker
import numpy as np

import self_improve_step
from polyglot.harness import harness as polyglot_harness
from prompts.self_improvement_prompt import find_selfimprove_eval_logs
from prompts.testrepo_prompt import get_test_description
from self_improve_step import diagnose_problem, save_metadata  # , choose
from swe_bench.harness import harness as swe_harness
from swe_bench.report import make_report
from utils.common_utils import load_json_file
from utils.docker_utils import (build_hgm_container, cleanup_container,
                                copy_from_container, copy_to_container,
                                log_container_output,
                                remove_existing_container, safe_log,
                                setup_logger)
from utils.eval_utils import get_acc_on_tasks
from utils.evo_utils import (get_all_performance, get_model_patch_paths,
                             is_compiled_self_improve)

dataset = None
alpha = 0.5
K = 0.5
bias_factor = 5
nodes = {}
total_tasks = []
output_dir = ""
polyglot = False
n_task_evals = 0
init_evaluated_tasks = []
llm = ""
timeout = 3600

pending_tasks_lock = threading.Lock()


def init(_polyglot, _output_dir, _tasks, _n_task_evals=0, _llm="", _timeout=3600):
    global output_dir, total_tasks, polyglot, n_task_evals, llm, timeout
    output_dir = _output_dir
    timeout = _timeout
    seen = set()
    total_tasks = []
    for item in _tasks:
        if item not in seen:
            seen.add(item)
            total_tasks.append(item)
    polyglot = _polyglot
    n_task_evals = _n_task_evals
    llm = _llm


def any_exceeding_context_length(output_dir, commit_id, instance_ids):
    """
    Check if any of the issues have exceeded the context length.
    """
    for instance_id in instance_ids:
        md_logs, _, _, _ = find_selfimprove_eval_logs(
            instance_id, output_dir, commit_id, filter=False
        )
        error_strs = [
            r"Error in get_response_withtools: Error code: 400 - {'message': 'Input is too long for requested model.'}",
            r"Error in get_response_withtools: Error code: 400 - {'object': 'error', 'message': \"This model's maximum context length is \d+ tokens. However, you requested \d+ tokens in the messages, Please reduce the length of the messages. None\", 'type': 'BadRequestError', 'param': None, 'code': 400}",
            r"Error in get_response_withtools: Error code: 400 - {'error': {'message': 'Your input exceeds the context window of this model. Please adjust your input and try again.', 'type': 'invalid_request_error', 'param': 'input', 'code': 'context_length_exceeded'}}",
        ]
        for md_log in md_logs:
            if any(
                re.search(f"{error_str}\n{error_str}", md_log)
                for error_str in error_strs
            ):
                return True
    return False


def choose_entry(parent_commit, debug=False):
    """
    Choose entry for self-improvement given a parent commit.
    """
    try:
        metadata_path = os.path.join(output_dir, parent_commit, "metadata.json")
        metadata = load_json_file(metadata_path)
        metadata = {
            "accuracy_score": metadata["overall_performance"]["accuracy_score"],
            "total_unresolved_ids": metadata["overall_performance"][
                "total_unresolved_ids"
            ],
            "total_emptypatch_ids": metadata["overall_performance"][
                "total_emptypatch_ids"
            ],
            "total_resolved_ids": metadata["overall_performance"]["total_resolved_ids"],
            "children_count": 0,
        }
    except Exception as e:
        # probably because swe-eval failed, generated code did not compile, etc.
        raise RuntimeError(f"{parent_commit} not eligible for being a parent: {e}")
    if debug:
        safe_log(metadata)

    empty_ids = metadata["total_emptypatch_ids"]
    resolved_ids = metadata["total_resolved_ids"]
    unresolved_ids = metadata["total_unresolved_ids"]

    entry = None

    if polyglot:
        entry_ids = empty_ids + unresolved_ids
        if not entry_ids:
            entry_ids = resolved_ids + empty_ids + unresolved_ids
        entry = random.choice(entry_ids)
    else:
        num_total_ids = len(empty_ids) + len(resolved_ids) + len(unresolved_ids)

        if len(empty_ids) >= 0.1 * num_total_ids and random.random() < 0.25:
            entry = "solve_empty_patches"

        elif random.random() < 0.25:
            entry = "solve_stochasticity"

        elif (
            any_exceeding_context_length(
                output_dir, parent_commit, empty_ids + unresolved_ids
            )
            and random.random() < 0.25
        ):
            entry = "solve_contextlength"

        elif len(unresolved_ids) != 0:
            entry_ids = unresolved_ids
            entry = random.choice(entry_ids)

        else:
            entry = random.choice(resolved_ids + empty_ids + unresolved_ids)
    if entry is None:
        safe_log(metadata)
        raise RuntimeError(
            f"Failed to choose an entry for self-improvement based on {parent_commit}."
        )
    return entry


def eval_agent(
    commit_id,
    tasks=None,
    num_tasks=5,
    max_workers=20,
    pending_tasks=None,
    random_level=0.5,
    skip=True,
    init_agent_path="./",
    config=None,
    recorder=None,
):
    if commit_id == "failed":
        return [0] * num_tasks
    global n_task_evals, total_tasks
    config = config if config is not None else HGMConfig()
    from config import validate_downstream_backend
    validate_downstream_backend(config.llm.downstream_backend)
    from candidate_provenance import (ROOT, ProvenanceError, read_chain,
                                      validate_metadata, validate_root)
    metadata_path = os.path.join(output_dir, commit_id, "metadata.json")
    if not os.path.exists(metadata_path):
        if commit_id != "initial":
            raise ProvenanceError("Candidate metadata is missing", code="PROV_ARTIFACT_UNREADABLE")
        metadata = {
                "candidate_schema_version": 1, "provenance": ROOT.to_dict(),
                "run_id": commit_id,
                "overall_performance": {
                    "accuracy_score": 0,
                    "total_resolved_instances": 0,
                    "total_submitted_instances": 0,
                    "files": [],
                    "total_unresolved_ids": [],
                    "total_emptypatch_ids": [],
                    "total_resolved_ids": [],
                    "total_submitted_ids": []
                }
            }
        save_metadata(metadata, os.path.join(output_dir, commit_id))
    metadata = load_json_file(metadata_path)
    if not isinstance(metadata, dict):
        raise ProvenanceError("Candidate metadata is not an object", code="PROV_METADATA_INVALID")
    node_dir = Path(output_dir).absolute() / commit_id
    initial_source = (commit_id == "initial" or
                      (metadata.get("run_id") == "initial" and (node_dir / "src").is_dir()
                       and Path(init_agent_path).resolve() == (node_dir / "src").resolve()))
    if initial_source:
        validate_root(node_dir)
        chain = ()
    else:
        chain = read_chain(Path(output_dir).absolute(), commit_id, require_versioned=True)
    patch_files = [str(Path(output_dir).absolute() / item.provenance.candidate_id / "model_patch.diff")
                   for item in chain]
    if tasks is None:
        if commit_id == "initial":
            if len(set(init_evaluated_tasks)) >= len(set(total_tasks)):
                return [metadata["overall_performance"]["accuracy_score"]] * num_tasks
            else:
                if skip:
                    un_evalatued_tasks = [
                        task for task in total_tasks if task not in init_evaluated_tasks
                    ]
                else:
                    un_evalatued_tasks = total_tasks
                order = "random" if random.random() < random_level else "fixed"
                if order == "random":
                    tasks = random.sample(
                        un_evalatued_tasks, min(num_tasks, len(un_evalatued_tasks))
                    )
                else:
                    tasks = un_evalatued_tasks[:num_tasks]
                init_evaluated_tasks.extend(tasks)
                return get_acc_on_tasks(tasks, os.path.join(output_dir, commit_id))
        if pending_tasks is None:
            pending_tasks = []

        with pending_tasks_lock:
            if skip:
                submitted_and_pending = (
                    metadata["overall_performance"]["total_submitted_ids"]
                    + pending_tasks
                )
                un_evalatued_tasks = [
                    task for task in total_tasks if task not in submitted_and_pending
                ]
            else:
                un_evalatued_tasks = total_tasks

            order = "random" if random.random() < random_level else "fixed"
            if len(un_evalatued_tasks) > 0:
                if order == "random":
                    tasks = random.sample(
                        un_evalatued_tasks, min(num_tasks, len(un_evalatued_tasks))
                    )
                else:
                    tasks = un_evalatued_tasks[:num_tasks]
                num_tasks = len(tasks)
            else:
                return [metadata["overall_performance"]["accuracy_score"]] * num_tasks
            pending_tasks.extend(tasks)

    from network_readiness import enabled as readiness_enabled, CompletionRecorder
    if readiness_enabled(config.codex) and config.llm.downstream_backend == "codex_exec":
        # The serial task boundary reports real completion (including ordinary
        # origin failures), but never a cancelled/timed-out readiness wait.
        def completed():
            global n_task_evals
            n_task_evals += 1
        recorder = CompletionRecorder(recorder, completed)
    else:
        n_task_evals += len(tasks)
    root_dir = os.path.abspath("./")
    metadata = load_json_file(
        os.path.join(root_dir, output_dir, commit_id, "metadata.json")
    )
    from run_metadata import file_hash
    metadata["evaluation_patch_sha256"] = [file_hash(p) for p in patch_files]
    downstream_options = dict(
        backend=config.llm.downstream_backend, codex_config=config.codex,
        evaluation_timeout=config.execution.evaluation_timeout,
        legacy_model=config.llm.downstream_llm,
        recorder=recorder,
    )
    if polyglot:
        polyglot_harness(
            test_task_list=tasks,
            max_workers=min(max_workers, len(tasks)),
            model_name_or_path=commit_id,
            model_patch_paths=patch_files,
            pred_dname=os.path.join(root_dir, output_dir, commit_id, "predictions"),
            output_dir=os.path.join(root_dir, output_dir, commit_id),
            init_agent_path=init_agent_path,
            dataset_path=config.evaluation.polyglot_dataset_path,
            **downstream_options,
        )
        overall_performance = get_all_performance(
            commit_id, results_dir=os.path.join(output_dir, commit_id)
        )[1]
    else:
        dnames = swe_harness(
            test_task_list=tasks,
            max_workers=min(max_workers, len(tasks)),
            model_name_or_path=commit_id,
            model_patch_paths=patch_files,
            pred_dname=os.path.join(root_dir, output_dir, commit_id, "predictions"),
            init_agent_path=init_agent_path,
            dataset_name=config.evaluation.dataset_name,
            dataset_path=config.evaluation.dataset_path,
            **downstream_options,
        )
        safe_log("Start make_report")
        make_report(
            dnames,
            run_ids=[f"{commit_id}_{i}" for i in range(len(dnames))],
            output_dir=os.path.join(output_dir, commit_id),
            num_eval_procs=min(max_workers, len(tasks)),
        )
        safe_log("Start get_performance")

        _, overall_performance = get_all_performance(
            commit_id, results_dir=os.path.join(output_dir, commit_id)
        )
        safe_log("End of evaluation")
        metadata["swe_dnames"] = [str(dn) for dn in dnames]

    metadata["overall_performance"] = overall_performance
    metadata["downstream_backend"] = config.llm.downstream_backend
    if not initial_source:
        chain = read_chain(os.path.join(root_dir, output_dir), commit_id, require_versioned=True)
        validate_metadata(metadata, commit_id, chain[-2].provenance if len(chain) > 1 else ROOT,
                          chain[-1].patch_bytes)
    save_metadata(metadata, os.path.join(root_dir, output_dir, commit_id))
    return get_acc_on_tasks(tasks, os.path.join(root_dir, output_dir, commit_id))


def sample_child(parent_commit, image_name, force_rebuild=False, max_try=1, *, diagnoser=None,
                 self_improve_backend="codex_exec", src_path=None, codex_config=None,
                 candidate_config=None, candidate_runner=None, diagnosis_dataset=None, recorder=None,
                 candidate_id=None):
    """Return a child run_id or 'failed'; max_try is the total attempt budget.

    As before, max_try <= 1 means one attempt. Retries always preserve named
    dependencies, especially image_name and force_rebuild. HGM owns scheduling.
    """
    validate_self_improve_backend(self_improve_backend)
    from candidate_provenance import require_candidate_backend, require_candidate
    require_candidate_backend(self_improve_backend)
    if candidate_id is not None and (self_improve_backend != "codex_exec" or max_try != 1):
        raise ValueError("Preassigned candidate IDs require one codex_exec attempt")
    if not isinstance(max_try, int) or isinstance(max_try, bool):
        raise ValueError("max_try must be an integer attempt budget")
    for attempt in range(max(1, max_try)):
        if self_improve_backend == "codex_exec":
            result = _sample_child_codex_once(
                parent_commit=parent_commit, src_path=src_path, diagnoser=diagnoser,
                codex_config=codex_config, candidate_config=candidate_config,
                candidate_runner=candidate_runner, diagnosis_dataset=diagnosis_dataset,
                recorder=recorder,
                **({"candidate_id": candidate_id} if candidate_id is not None else {}),
            )
        else:
            result = _sample_child_legacy_once(
                parent_commit=parent_commit, image_name=image_name,
                force_rebuild=force_rebuild, diagnoser=diagnoser,
            )
        if result != "failed":
            require_candidate(output_dir, result, parent_commit)
            return result
        safe_log(f"Child generation attempt {attempt + 1}/{max(1, max_try)} failed.")
    return "failed"


def _sample_child_codex_once(*, parent_commit, src_path, diagnoser, codex_config,
                            candidate_config, candidate_runner, diagnosis_dataset, recorder=None,
                            candidate_id=None):
    run_id = candidate_id or datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if candidate_id is not None and not re.fullmatch(r"candidate-[0-9]{6,}", candidate_id):
        raise ValueError("Invalid preassigned candidate ID")
    run_output_dir = os.path.join(os.path.abspath(output_dir), run_id)
    metadata = {"run_id": run_id, "parent_commit": parent_commit,
                "candidate_schema_version": 1, "artifact_type": "candidate_attempt", "provenance": None,
                "status": "failed", "backend": "codex_exec", "stage": "workspace"}
    owns_output = False
    try:
        os.makedirs(run_output_dir, exist_ok=False)
        owns_output = True
        if src_path is None or not os.path.isdir(src_path) or parent_commit == "failed":
            raise ValueError("Initial source is required")
        source = os.path.abspath(src_path)
        setup_logger(os.path.join(run_output_dir, "self_improve.log"))
        metadata["stage"] = "diagnosis"
        if diagnosis_dataset is None:
            if polyglot:
                diagnosis_dataset = load_json_file("polyglot/polyglot_benchmark_metadata.json")
            else:
                from downstream_solver import load_swe_dataset
                from config import load_config
                evaluation = load_config().evaluation
                diagnosis_dataset = load_swe_dataset(evaluation.dataset_name, evaluation.dataset_path)
        self_improve_step.dataset = diagnosis_dataset  # Existing diagnosis compatibility state.
        metadata["stage"] = "parent_patches"
        patch_files = get_model_patch_paths(os.path.abspath("."), output_dir, parent_commit, strict=True)
        metadata["stage"] = "diagnosis"
        entry = choose_entry(parent_commit)
        metadata["entry"] = entry  # Preserve the safe task/scenario ID even if diagnosis fails.
        diagnoser = diagnose_problem if diagnoser is None else diagnoser
        problem = diagnoser(entry, parent_commit, source, os.path.abspath(output_dir),
                            patch_files=patch_files, polyglot=polyglot)
        if not problem:
            raise ValueError("Diagnosis failed")
        metadata["stage"] = "candidate"
        result = generate_candidate(
            source, run_output_dir, patch_files, problem, get_test_description(swerepo=False),
            parent_commit=parent_commit, entry=entry, polyglot=polyglot,
            codex_config=codex_config, config=candidate_config, runner=candidate_runner,
            recorder=recorder,
        )
        if not result.ok:
            safe_log(f"Candidate failed at stage {result.metadata['stage']}: "
                     f"{result.metadata.get('reason', 'unknown')}")
        return run_id if result.ok else "failed"
    except Exception as error:
        metadata["reason"] = "Child preparation failed"
        from candidate_provenance import ProvenanceError
        if isinstance(error, ProvenanceError):
            metadata.update(reason=str(error), reason_code=error.code)
        safe_log(f"Child preparation failed at stage {metadata['stage']}.")
        if owns_output:
            try:
                save_metadata(metadata, run_output_dir)
            except OSError:
                safe_log("Could not save child failure metadata.")
        return "failed"


def _sample_child_legacy_once(*, parent_commit, image_name, force_rebuild=False, diagnoser=None):
    """Explicitly unsupported: the old container path lacks verified snapshots."""
    from candidate_provenance import require_candidate_backend
    require_candidate_backend("legacy_api")
