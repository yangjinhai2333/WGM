import argparse
import datetime
import json
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import docker
from datasets import load_dataset

from prompts.testrepo_prompt import get_test_description
from swe_bench.utils import (close_logger, copy_from_container, copy_to_container,
                             log_container_output, remove_existing_container,
                             safe_log, setup_logger)
from utils.common_utils import load_json_file
from config import validate_downstream_backend
from downstream_solver import (TaskFailure, container_command, put_bytes, run_task, image_facts,
                               load_agent_policy, load_swe_dataset, run_serial_tasks, select_entries)

llm = ""  # Global variable to hold the LLM model name or path
timeout = 1800
_image_build_lock = threading.Lock()


def _legacy_dependencies():
    global build_container, build_env_images, cleanup_container, make_test_spec
    from swebench.harness.docker_build import build_container, build_env_images, cleanup_container
    from swebench.harness.test_spec import make_test_spec


def _process_entry_legacy(
    entry, out_dname, model_name_or_path, model_patch_paths, init_agent_path="."
):
    """
    Process a single dataset entry. This function encapsulates the main processing logic
    for each entry to make it suitable for parallel execution.
    """
    _legacy_dependencies()
    instance_id = entry["instance_id"]
    problem_statement = entry["problem_statement"]
    base_commit = entry["base_commit"]
    chat_history_file = out_dname / (instance_id + ".md")
    out_fname = out_dname / (instance_id + ".json")
    eval_file = out_dname / f"{instance_id}_eval.sh"

    # Skip if output result already exists
    if out_fname.exists():
        print(f"Skipping existing entry {instance_id}")
        return {"success": True, "instance_id": instance_id}

    # Initialize container as None to avoid UnboundLocalError
    container = None

    try:
        # Create and start the Docker container
        client = docker.from_env()
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        # Set up thread-specific logger
        logger = setup_logger(str(out_dname / f"{instance_id}_docker.log"))
        nocache = True
        test_spec = make_test_spec(entry)
        # Remove any existing container with the same name
        container_name = test_spec.get_instance_container_name(run_id)
        remove_existing_container(client, container_name)
        # Now create and start the container
        container = build_container(
            test_spec, client, run_id, logger, nocache, force_rebuild=False
        )
        container.start()

        # Check that we are in hgm directory
        tmp_currdir = os.path.abspath(os.getcwd())
        logger.info(f"Current directory: {tmp_currdir}")
        # If not in hgm directory, try to change to it
        if "hgm" not in tmp_currdir.split("/")[-1].lower():
            try:
                os.chdir("hgm")
                tmp_currdir = os.path.abspath(os.getcwd())
                logger.info(f"Changed directory to: {tmp_currdir}")
            except Exception as e:
                pass
        # If still not in hgm directory, go up until we find it
        count = 0
        while "hgm" not in tmp_currdir.split("/")[-1].lower():
            os.chdir("..")
            tmp_currdir = os.path.abspath(os.getcwd())
            logger.info(f"Changed directory to: {tmp_currdir}")
            count += 1
            if count > 10:
                raise Exception("Could not find hgm directory after 10 attempts")

        # Copy the necessary files and requirements to the container
        copy_to_container(
            container,
            os.path.join(init_agent_path, "coding_agent.py"),
            "/hgm/coding_agent.py",
        )
        copy_to_container(
            container,
            os.path.join(init_agent_path, "requirements.txt"),
            "/hgm/requirements.txt",
        )
        if os.path.exists(os.path.join(init_agent_path, "pytest.ini")):
            copy_to_container(
                container, os.path.join(init_agent_path, "pytest.ini"), "/hgm/pytest.ini"
            )
        copy_to_container(
            container, os.path.join(init_agent_path, "tools/"), "/hgm/tools/"
        )
        copy_to_container(
            container, os.path.join(init_agent_path, "utils/"), "/hgm/utils/"
        )
        if os.path.isdir(os.path.join(init_agent_path, "tests")):
            copy_to_container(
                container, os.path.join(init_agent_path, "tests/"), "/hgm/tests/"
            )
        copy_to_container(
            container, os.path.join(init_agent_path, "prompts/"), "/hgm/prompts/"
        )
        copy_to_container(
            container, os.path.join(init_agent_path, "llm.py"), "/hgm/llm.py"
        )
        copy_to_container(
            container,
            os.path.join(init_agent_path, "llm_withtools.py"),
            "/hgm/llm_withtools.py",
        )
        copy_to_container(
            container, os.path.join(init_agent_path, "LICENSE"), "/hgm/LICENSE"
        )
        copy_to_container(
            container, os.path.join(init_agent_path, "README.md"), "/hgm/README.md"
        )
        chat_history_file_container = f"/hgm/{chat_history_file.name}"

        # Install issue requirements
        logger.info("Setting up environment")
        eval_script = test_spec.eval_script
        eval_file.write_text(eval_script)
        copy_to_container(container, eval_file, "/eval.sh")
        exec_result = container.exec_run(
            "/bin/bash /eval.sh", workdir="/"
        )  # setup environment
        log_container_output(exec_result)
        exec_result = container.exec_run(
            "rm /eval.sh", workdir="/"
        )  # remove eval script
        log_container_output(exec_result)

        # Get test description
        test_description = get_test_description(eval_script=eval_script, swerepo=True)

        # Apply model patch
        if model_patch_paths:
            safe_log("Applying model patches")
            for model_patch_path in model_patch_paths:
                copy_to_container(container, model_patch_path, "/hgm/parent_patch.txt")
                exec_result = container.exec_run(
                    "/bin/sh -c 'patch -p1 < /hgm/parent_patch.txt'", workdir="/hgm"
                )
                log_container_output(exec_result)
                exec_result = container.exec_run(
                    "rm /hgm/parent_patch.txt", workdir="/hgm"
                )
                log_container_output(exec_result)

        # Install this repo requirements
        safe_log("Installing more requirements")
        exec_result = container.exec_run(
            "python -m pip install -r /hgm/requirements.txt", workdir="/"
        )
        log_container_output(exec_result)

        # Run the agent
        env_vars = {
            "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
            "AWS_REGION": os.getenv("AWS_REGION"),
            "AWS_REGION_NAME": os.getenv("AWS_REGION_NAME"),
            "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID"),
            "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY"),
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "OpenRouter_API_KEY": os.getenv("OpenRouter_API_KEY"),
        }
        safe_log("Running the agent")
        cmd = [
            "timeout",
            str(timeout),  # 30min timeout
            "python",
            "/hgm/coding_agent.py",
            "--problem_statement",
            problem_statement,
            "--git_dir",
            "/testbed/",
            "--chat_history_file",
            chat_history_file_container,
            "--base_commit",
            base_commit,
            "--outdir",
            "/hgm/",
            "--test_description",
            test_description,
            "--instance_id",
            instance_id,
            "--model",
            llm,  # Use the global llm variable
            "--timeout",
            str(timeout),
        ]
        exec_result = container.exec_run(cmd, environment=env_vars, workdir="/")
        log_container_output(exec_result)

        # Copy output files back to host
        logger.info("Copying output files back to host")
        copy_from_container(container, chat_history_file_container, chat_history_file)
        # Additional chat history files
        exec_result = container.exec_run(
            f"find /hgm/ -name '{instance_id}_*.md'", workdir="/"
        )
        chat_history_files_container = exec_result.output.decode().split()
        for chat_history_file_container in chat_history_files_container:
            chat_history_file = out_dname / Path(chat_history_file_container).name
            copy_from_container(
                container, chat_history_file_container, chat_history_file
            )

        # Get model_patch
        logger.info("Getting model_patch")
        exec_result = container.exec_run("cat /hgm/model_patch.diff")
        log_container_output(exec_result)
        model_patch = exec_result.output.decode()
        # Additional proposed model patches
        proposed_model_patches = []
        exec_result = container.exec_run(
            "find /hgm/ -name 'model_patch_*.diff'", workdir="/"
        )
        model_patch_files_container = exec_result.output.decode().split()
        for model_patch_file_container in model_patch_files_container:
            exec_result = container.exec_run(f"cat {model_patch_file_container}")
            log_container_output(exec_result)
            proposed_model_patch = exec_result.output.decode()
            proposed_model_patches.append(proposed_model_patch)

        # Write result to file
        result = {
            "instance_id": instance_id,
            "model_name_or_path": model_name_or_path,
            "model_patch": model_patch,
            "proposed_model_patches": proposed_model_patches,
        }
        out_fname.write_text(json.dumps(result, indent=4))

        return {"success": True, "instance_id": instance_id}

    except Exception as e:
        print(f"Error processing entry {instance_id}: {str(e)}")
        # Create a result file with empty model_patch to mark it as submitted
        try:
            result = {
                "instance_id": instance_id,
                "model_name_or_path": model_name_or_path,
                "model_patch": "",  # Empty patch for failed cases
                "proposed_model_patches": [],
            }
            out_fname.write_text(json.dumps(result, indent=4))
        except Exception as write_error:
            print(f"Error writing empty result for {instance_id}: {str(write_error)}")
        return {"success": False, "instance_id": instance_id, "error": str(e)}

    finally:
        # Clean up docker container
        try:
            if container is not None:
                cleanup_container(client, container, logger)
        except Exception as e:
            print(f"Error cleaning up Docker container for {instance_id}: {e}")


def _harness_legacy(
    test_task_list=None,
    max_workers=4,
    model_name_or_path=None,
    model_patch_paths=None,
    pred_dname="./swe_bench/predictions",
    init_agent_path=".",
):
    """
    Parallel processing harness using ThreadPoolExecutor.

    Args:
        test_task_list: List of task IDs to process (None for all)
        num_samples: Number of samples to process (-1 for all)
        max_workers: Maximum number of concurrent threads
        model_name_or_path: Model name or path
        model_patch_paths: Paths to the model patches for hgm
        num_evals: Repeated number of swe evaluations
    """
    _legacy_dependencies()
    # Load dataset
    dataset = load_dataset("princeton-nlp/SWE-bench")
    dataset = dataset["test"]

    # Ensure that necessary directories exist
    if model_name_or_path is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name_or_path = f"{timestamp}--agent"
    pred_dname = Path(pred_dname)
    pred_dname.mkdir(exist_ok=True)

    # Prepare the dataset entries
    entries = list(dataset)
    entries = [entry for entry in entries if entry["instance_id"] in test_task_list]

    # Build the environment images
    client = docker.from_env()
    build_env_images(
        client, dataset=entries, force_rebuild=False, max_workers=max_workers
    )
    # Get all subdirectories under pred_dname
    subdirs = [d for d in pred_dname.iterdir() if d.is_dir()]
    # Define a function to handle a single evaluation for all specified issues
    model_name_or_path_inst = f"{model_name_or_path}_{len([d for d in subdirs if d.name.startswith(model_name_or_path)])+1}"
    out_dname = pred_dname / model_name_or_path_inst
    out_dname.mkdir(exist_ok=True)

    print(f"Starting evaluation {0} for model {model_name_or_path}")

    # Process entries in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_entry = {
            executor.submit(
                _process_entry_legacy,
                entry,
                out_dname,
                model_name_or_path_inst,
                model_patch_paths,
                init_agent_path,
            ): entry
            for entry in entries
        }

        # Process completed tasks as they finish
        for future in as_completed(future_to_entry):
            result = future.result()
            if result["success"]:
                print(
                    f"Successfully processed entry {result['instance_id']} for eval {0}"
                )
            else:
                print(
                    f"Failed to process entry {result['instance_id']} for eval {0}: {result.get('error', 'Unknown error')}"
                )

    # Process all evaluations in parallel
    print(f"All evaluations completed for model {model_name_or_path}")
    return [out_dname]


def _codex_callbacks(entry, evaluation_timeout, *, ledger_enabled=False, evaluation_id=None):
    """Keep the pinned SWE-bench image, eval script and grading semantics."""
    state = {}

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
        from swebench.harness.docker_build import build_container, build_env_images
        from swebench.harness.test_spec import make_test_spec
        client = state["client"] = docker.from_env(timeout=evaluation_timeout + 60)
        container = None
        try:
            spec = state["spec"] = make_test_spec(entry)
            logger = state["logger"] = setup_logger(str(log_dir / "docker.log"))
            try:
                client.images.get(spec.instance_image_key)
                cached = True
            except docker.errors.ImageNotFound:
                cached = False
            # SWE image tags/build logs are shared; only preparation is serialized.
            # Each validated evaluation owns its container and writable testbed.
            with _image_build_lock:
                _, failed = build_env_images(client, dataset=[entry], force_rebuild=False, max_workers=1)
                if failed:
                    raise TaskFailure("SWE environment image preparation failed")
                container = build_container(spec, client, evaluation_id or datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
                                            logger, False, force_rebuild=False)
            container.start()
            description = get_test_description(eval_script=spec.eval_script, swerepo=True)
            facts = image_facts(client, spec)
            if ledger_enabled:
                from evaluation_ledger import digest
                prepare.testspec_fingerprint = digest({"task_id":entry["instance_id"], "script":spec.eval_script,
                                                       "image":spec.instance_image_key})
                prepare.image_fingerprint = digest(facts)
            return container, description, cached, facts
        except BaseException:
            if container is not None:
                cleanup(container)
            else:
                client.close()
                if "logger" in state:
                    close_logger(state["logger"])
                prepare.cleanup_verified = True
            raise

    def evaluate(container, prediction, log_dir):
        from swebench.harness.grading import get_eval_report
        spec = state["spec"]
        put_bytes(container, "/eval.sh", spec.eval_script.encode())
        result = container_command(container, ["timeout", "--kill-after=10s", str(evaluation_timeout),
                                               "/bin/bash", "/eval.sh"], check=False)
        output = result.output.decode("utf-8", errors="replace")
        log_path = log_dir / "test_output.txt"
        log_path.write_text(output)
        expired = result.exit_code in (124, 137)
        if expired:
            return {"resolved": False, "timed_out": True, "returncode": result.exit_code}
        report = get_eval_report(spec, prediction, log_path, include_tests_status=True)
        (log_dir / "report.json").write_text(json.dumps(report, indent=2))
        graded = report[entry["instance_id"]]
        return {"resolved": graded["resolved"], "timed_out": False,
                "returncode": result.exit_code,
                "error": not graded["patch_successfully_applied"]}

    def restore(resource_id, log_dir, *, cleanup=False):
        # Only look up this attempt's owned resource; never create/start one.
        from evaluation_ledger import digest
        client = state.get("client")
        if client is None:
            client = state["client"] = docker.from_env(timeout=evaluation_timeout + 60)
        container = client.containers.get(resource_id)
        if cleanup:
            return container, ""
        from swebench.harness.test_spec import make_test_spec
        spec = state.get("spec")
        if spec is None:
            spec = state["spec"] = make_test_spec(entry)
        prepare.testspec_fingerprint = digest({"task_id":entry["instance_id"], "script":spec.eval_script,
                                               "image":spec.instance_image_key})
        prepare.image_fingerprint = digest(image_facts(client, spec))
        return container, get_test_description(eval_script=spec.eval_script, swerepo=True)

    def lifecycle_evaluate(container, prediction, log_dir, *, on_started):
        from downstream_lifecycle import container_evaluator_result
        from swebench.harness.grading import get_eval_report
        return container_evaluator_result(container, prediction, state["spec"], evaluation_timeout,
                                          get_eval_report, on_started, log_root=log_dir)

    import hashlib
    prepare.restore = restore
    evaluate.lifecycle = lifecycle_evaluate
    evaluate.evaluator_fingerprint = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return prepare, evaluate, cleanup


def process_entry(entry, out_dname, model_name_or_path, model_patch_paths=None,
                  init_agent_path=".", *, backend="codex_exec", codex_config=None,
                  evaluation_timeout=None, runner=None, policy=None,
                  prepare=None, evaluate=None, cleanup=None, recorder=None,
                  execution_spec=None, evaluation_ledger=None, source_fingerprint=None):
    validate_downstream_backend(backend)
    ledger_options = {}
    if execution_spec is not None or evaluation_ledger is not None:
        if backend != "codex_exec":
            raise ValueError("Evaluation lifecycle requires codex_exec")
        if policy is None or source_fingerprint is None:
            raise ValueError("Evaluation lifecycle requires verified policy and source identity")
        ledger_options = dict(execution_spec=execution_spec, evaluation_ledger=evaluation_ledger,
                              source_fingerprint=source_fingerprint)
    if backend == "legacy_api":
        return _process_entry_legacy(entry, out_dname, model_name_or_path, model_patch_paths, init_agent_path)
    from swe_bench.report import remove_patches_to_tests
    limit = evaluation_timeout if evaluation_timeout is not None else timeout
    if prepare is None:
        if ledger_options:
            prepare, evaluate, cleanup = _codex_callbacks(entry, limit, ledger_enabled=True,
                                                        evaluation_id=execution_spec.identity.evaluation_id)
        else:
            prepare, evaluate, cleanup = _codex_callbacks(entry, limit)
    if policy is None:
        policy = load_agent_policy(init_agent_path, model_patch_paths)
    return run_task(entry, out_dname, model_name_or_path, prepare=prepare,
                    evaluate=evaluate, cleanup=cleanup, runner=runner,
                    codex_config=codex_config, timeout=limit, policy=policy,
                    patch_filter=remove_patches_to_tests, recorder=recorder, **ledger_options)


def harness(test_task_list=None, max_workers=4, model_name_or_path=None,
            model_patch_paths=None, pred_dname="./swe_bench/predictions", init_agent_path=".",
            *, backend="codex_exec", codex_config=None, evaluation_timeout=None,
            dataset_name="princeton-nlp/SWE-bench", dataset_path="swe_bench/dataset",
            n_tasks=None, dataset=None,
            runner=None, resume_dir=None, legacy_model=None, recorder=None):
    validate_downstream_backend(backend)
    if backend == "legacy_api":
        global llm, timeout
        if legacy_model is not None:
            llm = legacy_model
        if evaluation_timeout is not None:
            timeout = evaluation_timeout
        return _harness_legacy(test_task_list, max_workers, model_name_or_path,
                               model_patch_paths, pred_dname, init_agent_path)
    started = time.monotonic()
    if dataset is None:
        dataset = load_swe_dataset(dataset_name, dataset_path)
    entries = select_entries(dataset, test_task_list, n_tasks)
    policy = load_agent_policy(init_agent_path, model_patch_paths)
    model_name_or_path = model_name_or_path or "codex"
    out_dir = Path(resume_dir) if resume_dir else Path(pred_dname) / (
        model_name_or_path + "_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    if not resume_dir:
        out_dir.mkdir(parents=True, exist_ok=False)
    preparation = time.monotonic() - started
    def process(entry, directory):
        return process_entry(entry, directory, model_name_or_path, backend=backend,
                             codex_config=codex_config, evaluation_timeout=evaluation_timeout,
                             runner=runner, policy=policy, recorder=recorder)
    return run_serial_tasks(entries, out_dir, process, total_instances=len(dataset),
                            configured_workers=max_workers, preparation_seconds=preparation)


def _legacy_main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_samples", type=int, default=-1, help="Number of samples to process"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Maximum number of concurrent threads",
    )
    parser.add_argument(
        "--model_name_or_path", type=str, default=None, help="Model name or path"
    )
    parser.add_argument(
        "--model_patch_paths", type=str, default=None, help="Paths to the model patches"
    )
    parser.add_argument(
        "--num_evals", type=int, default=1, help="Repeated number of swe evaluations"
    )
    parser.add_argument(
        "--num_evals_parallel",
        type=int,
        default=1,
        help="Number of parallel repeated evaluations",
    )
    parser.add_argument(
        "--pred_dname",
        type=str,
        default="./swe_bench/predictions",
        help="Output directory for predictions",
    )
    parser.add_argument(
        "--test_task_list",
        type=str,
        default=None,
        help="Subset of swe issues to process",
    )
    args = parser.parse_args()

    # Load the test task list
    if args.test_task_list == "small":
        test_task_list = load_json_file("./swe_bench/subsets/small.json")
    elif args.test_task_list == "medium":
        test_task_list = load_json_file("./swe_bench/subsets/medium.json")
    else:
        # Default to None, means all of them
        test_task_list = None

    # Run the parallel harness
    harness(
        test_task_list=test_task_list,
        num_samples=args.num_samples,
        max_workers=args.max_workers,
        model_name_or_path=args.model_name_or_path,
        model_patch_paths=args.model_patch_paths,
        num_evals=args.num_evals,
        num_evals_parallel=args.num_evals_parallel,
        pred_dname=args.pred_dname,
    )


if __name__ == "__main__":
    from pipeline import main
    main()
