import argparse
import datetime
import json
import math
import os
import random
import shutil
import signal
import string
import threading
import time
import traceback
from collections import defaultdict
from concurrent.futures import (ProcessPoolExecutor, ThreadPoolExecutor,
                                TimeoutError, as_completed)
from statistics import stdev
from functools import partial

import numpy as np
from datasets import load_dataset
from utils.docker_utils import copy_src_files

import hgm_utils
from config import effective_worker_count, load_config
from pipeline import require_preflight
from run_metadata import build_manifest, open_manifest, RunRecorder
from downstream_solver import load_swe_dataset
from tree import Node
from utils.common_utils import load_json_file
from utils.docker_utils import copy_src_files, setup_logger
from utils.evo_utils import load_hgm_metadata
from search_config import is_corrected, optimization as profile_optimization
from root_cache import prepare_search_root, verify_imported_root
from final_evaluation import run_final_evaluation
from protocol_config import configure_search_protocol
from task_schedule import ordered_tasks
from search_state import (SearchState, best_belief, save_checkpoint, load_checkpoint,
                          validate_resume_profile, lock_run)
from run_metadata import atomic_json, finalize_search_manifest


def update_metadata(output_dir, n_task_evals):
    with open(os.path.join(output_dir, "hgm_metadata.jsonl"), "a") as f:
        f.write(
            json.dumps(
                {
                    "n_task_evals": n_task_evals,
                    "nodes": [
                        node.save_as_dict()
                        for node in hgm_utils.nodes.values()
                        if node.commit_id != "initial"
                    ],
                },
                indent=2,
            )
            + "\n"
        )
    json.dump(
        hgm_utils.init_evaluated_tasks,
        open(os.path.join(output_dir, "init_evaluated_tasks.json"), "w"),
    )


def initialize_run(
    output_dir,
    self_improve_llm,
    tasks,
    initial_agent_name,
    prevrun_dir=None,
    polyglot=False,
    timeout=3600,
    max_workers=20,
    config=None,
    recorder=None,
    verified_initial=None,
):
    from candidate_provenance import require_candidate_backend, validate_root, validate_search_nodes
    require_candidate_backend(config.llm.self_improve_backend if config is not None else "codex_exec")
    hgm_utils.init(polyglot, output_dir, tasks, 0, self_improve_llm, timeout)

    # Copy cached initial version into experiment dir
    initial_folder = "initial_swe/" if not polyglot else "initial_polyglot/"
    if not prevrun_dir and verified_initial is None:
        if not os.path.exists(f"{initial_folder}/{initial_agent_name}"):
            copy_src_files(f"{initial_folder}/{initial_agent_name}/src",
                           build_image=config is not None and config.llm.self_improve_backend == "legacy_api")
            hgm_utils.output_dir = initial_folder
            hgm_utils.eval_agent(
                initial_agent_name,
                tasks = hgm_utils.total_tasks,
                max_workers=max_workers,
                init_agent_path=f"{initial_folder}/{initial_agent_name}/src",
                config=config,
                recorder=recorder,
            )
            hgm_utils.output_dir = output_dir

    if not prevrun_dir:
        initial_input = str(verified_initial) if verified_initial is not None else f"{initial_folder}/{initial_agent_name}"
        cached = load_json_file(os.path.join(initial_input, "metadata.json"))
        if config is not None and cached.get("downstream_backend") != config.llm.downstream_backend:
            raise ValueError("Cached initial evaluation uses a different backend; choose a new initial_agent_name")
        shutil.copytree(initial_input, os.path.join(output_dir, "initial"))
        if verified_initial is not None:
            verify_imported_root(initial_input, os.path.join(output_dir, "initial"),
                                 recorder.manifest.get("root_initialization") if recorder is not None else None)

    validate_root(os.path.join(output_dir, "initial"))
    Node(commit_id="initial")
    if prevrun_dir:
        # Load previous run's archive
        hgm_utils.init_evaluated_tasks = load_json_file(
            os.path.join(prevrun_dir, "init_evaluated_tasks.json")
        )
        metadata_path = os.path.join(prevrun_dir, "hgm_metadata.jsonl")
        metadata = load_hgm_metadata(metadata_path, last_only=True)
        validate_search_nodes(output_dir, [
            {"id": 0, "commit_id": "initial", "parent_id": None}, *metadata["nodes"]])
        for node in metadata["nodes"]:
            commit_id = node["commit_id"]
            parent_id = node["parent_id"]
            Node(commit_id, parent_id=parent_id, id=node["id"])
        for node in hgm_utils.nodes.values():
            if node.parent_id is not None:
                parent = hgm_utils.nodes[node.parent_id]
                parent.add_child(node)

    n_task_evals = 0
    submitted_ids = defaultdict(set)  # node_id -> set of submitted task ids
    for node in hgm_utils.nodes.values():
        metadata = load_json_file(
            os.path.join(output_dir, node.commit_id, "metadata.json")
        )
        submitted_ids[node.id] = set(
            metadata["overall_performance"]["total_submitted_ids"]
        )
        node.utility_measures = [
            1
            for _ in range(metadata["overall_performance"]["total_resolved_instances"])
        ] + [
            0
            for _ in range(
                metadata["overall_performance"]["total_submitted_instances"]
                - metadata["overall_performance"]["total_resolved_instances"]
            )
        ]
        if config is not None and is_corrected(config):
            # Own task outcomes only; preserve the cached task order.
            performance = metadata["overall_performance"]
            node.utility_measures = [int(task in performance["total_resolved_ids"])
                                     for task in dict.fromkeys(performance["total_submitted_ids"])]
            node.completed_task_results = {task: int(task in performance["total_resolved_ids"])
                                           for task in dict.fromkeys(performance["total_submitted_ids"])}
        if node.commit_id != "initial":
            n_task_evals += metadata["overall_performance"]["total_submitted_instances"]
    hgm_utils.n_task_evals = n_task_evals
    return (os.path.join(output_dir, "initial", "src") if verified_initial is not None else
            os.path.join(initial_folder, initial_agent_name, "src")), submitted_ids


def main():
    run_started = time.monotonic()
    readiness_cancel = None
    def interrupted(signum, frame):
        if readiness_cancel is not None:
            readiness_cancel.set()
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description="Optimistic Tree Search")
    from network_readiness import add_cli, configure_cli, safe_configuration, ReadinessAbort, cancellable_operation
    add_cli(parser)
    parser.add_argument("--algorithm", choices=("hgm", "wgm", "wgm_v2", "dgm"))
    parser.add_argument("--dgm-profile", choices=("legacy_code",))
    parser.add_argument("--dgm-max-generations", type=int)
    parser.add_argument("--dgm-shallow-eval", action="store_true", default=None)
    parser.add_argument("--smoke-target-budget", type=int, help="V2 smoke pause point (default 8); horizon remains 800")
    parser.add_argument("--target-budget", type=int,
                        help="Incremental HGM / WGM V2 / DGM pause target; planning horizon remains 800")
    parser.add_argument("--hgm_profile", choices=("legacy_code", "paper_corrected"))
    parser.add_argument("--execution-policy", choices=("origin", "validated"), default=None,
                        help="Override execution.policy in config; defaults to origin when neither is set")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--codex-model", help="Explicit model for this run; does not edit config.yaml")
    parser.add_argument("--reasoning-effort", choices=("minimal", "low", "medium", "high", "xhigh", "max"))
    parser.add_argument("--task-order-seed", type=int)
    parser.add_argument("--root-cache-dir")
    parser.add_argument("--initial-baseline-dir")
    parser.add_argument("--protocol", choices=("formal", "pilot", "smoke"),
                        help="pilot defaults to 20 search evaluations and requires all 60 final tasks")
    parser.add_argument("--smoke", action="store_true", default=None)
    parser.add_argument("--allow-legacy-initial-cache", action="store_true", default=None)
    parser.add_argument("--final-eval-task-limit", type=int)
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--max_task_evals",
        type=int,
        default=None,
        help="Search task-evaluation budget: pilot defaults to 20; formal requires 800.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=None,
        help="Number of parallel workers for self-improvement attempts.",
    )
    parser.add_argument("--evaluation-workers", type=int, default=None,
                        help="Fixed validated evaluations only; adaptive search remains serial")
    parser.add_argument(
        "--continue_from",
        type=str,
        default=None,
        help="Directory to continue the run from.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for this run (overrides config).",
    )
    parser.add_argument(
        "--polyglot",
        dest="polyglot",
        action="store_true",
        help="Run Polyglot benchmark instead of SWE-bench.",
    )
    parser.add_argument(
        "--no_polyglot",
        dest="polyglot",
        action="store_false",
        help="Disable Polyglot benchmark even if enabled in config.",
    )
    parser.add_argument(
        "--self_improve_llm",
        type=str,
        default=None,
        help="LLM model used only by the legacy_api self-improvement backend",
    )
    parser.add_argument(
        "--downstream_llm",
        type=str,
        default=None,
        help="LLM model to use for downstream tasks",
    )
    parser.add_argument(
        "--diagnose_llm",
        type=str,
        default=None,
        help="LLM model used only by the legacy_api diagnosis backend",
    )
    parser.add_argument(
        "--alpha", type=float, default=None, help="Alpha parameter for node expansion."
    )
    parser.add_argument(
        "--cool_down",
        dest="cool_down",
        action="store_true",
        help="Use a decreasing temperature over iterations.",
    )
    parser.add_argument(
        "--no_cool_down",
        dest="cool_down",
        action="store_false",
        help="Disable decreasing temperature over iterations even if enabled in config.",
    )
    parser.add_argument(
        "--beta", type=float, default=None, help="Cooling down factor beta."
    )
    parser.add_argument(
        "--full_eval",
        dest="full_eval",
        action="store_true",
        help="Run full evaluation on SWE even if disabled in config.",
    )

    parser.add_argument(
        "--self_improve_timeout",
        type=int,
        default=None,
        help="Timeout for self-improvement attempts.",
    )
    parser.add_argument(
        "--evaluation_timeout",
        type=int,
        default=None,
        help="Timeout for evaluation attempts.",
    )
    parser.add_argument(
        "--n_pseudo_descendant_evals",
        type=int,
        default=None,
        help="Number of pseudo descendant evaluations.",
    )
    parser.add_argument(
        "--eval_random_level",
        type=float,
        default=None,
        help="Randomness level for evaluation task selection.",
    )
    parser.add_argument(
        "--initial_agent_name",
        type=str,
        default=None,
        help="Initial agent name; V2 honors YAML, other algorithms retain default_agent.",
    )

    parser.set_defaults(polyglot=None, cool_down=None, full_eval=None)

    args = parser.parse_args()

    overrides = {}
    if args.max_task_evals is not None:
        overrides["execution.max_task_evals"] = args.max_task_evals
    if args.max_workers is not None:
        overrides["execution.max_workers"] = args.max_workers
    if args.evaluation_workers is not None:
        overrides["execution.evaluation_workers"] = args.evaluation_workers
    if args.continue_from is not None:
        overrides["paths.continue_from"] = args.continue_from
    if args.output_dir is not None:
        overrides["paths.output_dir"] = args.output_dir
    if args.self_improve_llm is not None:
        overrides["llm.self_improve_llm"] = args.self_improve_llm
    if args.downstream_llm is not None:
        overrides["llm.downstream_llm"] = args.downstream_llm
    if args.diagnose_llm is not None:
        overrides["llm.diagnose_llm"] = args.diagnose_llm
    if args.alpha is not None:
        overrides["optimization.alpha"] = args.alpha
    if args.cool_down is not None:
        overrides["optimization.cool_down"] = args.cool_down
    if args.beta is not None:
        overrides["optimization.beta"] = args.beta
    if args.full_eval is not None:
        overrides["evaluation.full_eval"] = args.full_eval
    if args.self_improve_timeout is not None:
        overrides["execution.self_improve_timeout"] = args.self_improve_timeout
    if args.evaluation_timeout is not None:
        overrides["execution.evaluation_timeout"] = args.evaluation_timeout
    if args.n_pseudo_descendant_evals is not None:
        overrides["optimization.n_pseudo_descendant_evals"] = args.n_pseudo_descendant_evals
    if args.eval_random_level is not None:
        overrides["optimization.eval_random_level"] = args.eval_random_level
    if args.polyglot is not None:
        overrides["evaluation.polyglot"] = args.polyglot
    if args.initial_agent_name is not None:
        overrides["paths.initial_agent_name"] = args.initial_agent_name

    config = load_config(args.config, **overrides)
    for key, value in (("search.algorithm", args.algorithm), ("search.hgm_profile", args.hgm_profile),
                       ("search.seed", args.seed),
                       ("search.dgm_profile", args.dgm_profile),
                       ("dgm.max_generations", args.dgm_max_generations),
                       ("dgm.shallow_eval", args.dgm_shallow_eval),
                       ("wgm_v2.smoke_target_budget", args.smoke_target_budget),
                       ("codex.model", args.codex_model),
                       ("codex.reasoning_effort", args.reasoning_effort),
                       ("evaluation.task_order_seed", args.task_order_seed),
                       ("paths.root_cache_dir", args.root_cache_dir), ("paths.initial_baseline_dir", args.initial_baseline_dir),
                       ("protocol.name", args.protocol),
                       ("protocol.smoke", args.smoke), ("protocol.allow_legacy_initial_cache", args.allow_legacy_initial_cache),
                       ("protocol.final_eval_task_limit", args.final_eval_task_limit)):
        if value is not None:
            config._set_nested_attr(key, value)
    if args.target_budget is not None:
        if config.search.algorithm == "hgm":
            from hgm_incremental_config import enabled
            if not enabled(config):
                parser.error("HGM --target-budget requires opt-in hgm_incremental configuration")
            section = "hgm_incremental"
        elif config.search.algorithm in ("dgm", "wgm_v2"):
            section = config.search.algorithm
        else:
            parser.error("--target-budget is unsupported for this algorithm")
        config._set_nested_attr(section + ".target_budget", args.target_budget)
    if args.initial_agent_name is None and config.search.algorithm != "wgm_v2":
        # Preserve the historical non-V2 CLI default, including its YAML precedence.
        config.paths.initial_agent_name = "default_agent"
    config.validate_search()
    from candidate_provenance import (ProvenanceError, require_candidate_backend,
                                      require_candidate, validate_search_nodes)
    require_candidate_backend(config.llm.self_improve_backend)
    configure_search_protocol(config, args.max_task_evals)
    from execution_policy import configure, validated, check_resume
    configure(config, args.execution_policy)
    configure_cli(config, args)
    if config.network_readiness.enabled:
        from threading import Event
        readiness_cancel = Event()
        config.codex.readiness_cancel_event = readiness_cancel
        signal.signal(signal.SIGINT, interrupted)
    if config.search.algorithm == "wgm_v2":
        from wgm_v2_search import run_search
        return run_search(config, preflight=require_preflight)
    if config.search.algorithm == "dgm":
        from dgm_legacy_search import run_search
        return run_search(config, preflight=require_preflight)
    if config.paths.continue_from:
        check_resume(config.paths.continue_from, config)
    if validated(config):
        from validated_runs import run_search
        return run_search(config, preflight=require_preflight)
    corrected = is_corrected(config)
    if corrected and any(backend != "codex_exec" for backend in (
            config.llm.self_improve_backend, config.llm.downstream_backend, config.llm.diagnose_backend)):
        raise ValueError("paper_corrected/WGM v0.1 requires serial codex_exec backends")
    effective_branch_count = None
    if config.paths.continue_from:
        effective_branch_count = validate_resume_profile(config.paths.continue_from, config)
        if config.paths.output_dir and os.path.abspath(config.paths.output_dir) != os.path.abspath(config.paths.continue_from):
            raise ValueError("Resume must use its original output directory")
        config.paths.output_dir = config.paths.continue_from
    require_preflight(config, output_root=config.paths.output_dir)

    if not config.paths.initial_agent_name and not (corrected and (config.paths.root_cache_dir or config.paths.initial_baseline_dir)):
        parser.error(
            "Initial agent name must be provided either in config.yaml or via --initial_agent_name."
        )

    llm_cfg = config.llm
    opt_cfg = profile_optimization(config)
    exec_cfg = config.execution
    eval_cfg = config.evaluation
    path_cfg = config.paths
    effective_workers = effective_worker_count(exec_cfg.max_workers, llm_cfg.self_improve_backend,
                                               llm_cfg.downstream_backend, llm_cfg.diagnose_backend)

    # Variables for this HGM run
    if path_cfg.output_dir:
        output_dir = os.path.abspath(path_cfg.output_dir)
        run_id = os.path.basename(os.path.normpath(output_dir))
    elif not path_cfg.continue_from:
        run_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S_%f")
        output_dir = os.path.abspath(os.path.join("./output_hgm", run_id))
    else:
        run_id = os.path.basename(os.path.normpath(path_cfg.continue_from))
        output_dir = os.path.abspath(os.path.join("./output_hgm", run_id))

    # Ensure output directory exists and log path info
    if corrected and not path_cfg.continue_from and os.path.exists(output_dir):
        raise FileExistsError("Corrected search requires a new output directory")
    os.makedirs(output_dir, exist_ok=True)
    run_lease = lock_run(output_dir) if corrected else None
    print(f"Working directory: {os.getcwd()}")
    print(f"Using config file: {args.config}")
    print(f"Output directory: {output_dir}")
    print(f"Output directory exists: {os.path.exists(output_dir)}")

    import polyglot.harness
    import self_improve_step

    polyglot.harness.llm = llm_cfg.downstream_llm  # Set the LLM model for downstream tasks
    import swe_bench.harness

    swe_bench.harness.llm = (
        llm_cfg.downstream_llm
    )  # Set the LLM model for downstream tasks
    polyglot.harness.timeout = exec_cfg.evaluation_timeout
    swe_bench.harness.timeout = exec_cfg.evaluation_timeout
    self_improve_step.self_improve_llm = llm_cfg.self_improve_llm
    # Initialize logger early
    logger = setup_logger(os.path.join(output_dir, "hgm_outer.log"))
    logger.info(f"Workers: configured={exec_cfg.max_workers}, effective={effective_workers}, "
                f"self_improve_backend={llm_cfg.self_improve_backend}, downstream_backend={llm_cfg.downstream_backend}")
    # SWE issues to consider
    if not eval_cfg.polyglot:
        if eval_cfg.full_eval:
            tasks = [
                task["instance_id"]
                for task in load_swe_dataset(eval_cfg.dataset_name, eval_cfg.dataset_path)
            ]
        else:
            tasks = load_json_file("./swe_bench/subsets/small.json") \
                    + load_json_file("./swe_bench/subsets/medium.json") 
        if corrected:
            tasks = ordered_tasks(tasks, config.evaluation.task_order_seed)
            random.seed(config.search.seed)
        else:
            random.seed(42)
            random.shuffle(tasks)
    else:
        tasks = load_json_file("./polyglot/subsets/medium.json") + load_json_file(
            "./polyglot/subsets/small.json"
        )
        if corrected:
            random.seed(config.search.seed)
    if corrected:
        np.random.seed(config.search.seed)

    diagnosis_dataset = (load_json_file(eval_cfg.polyglot_dataset_path) if eval_cfg.polyglot
                         else load_swe_dataset(eval_cfg.dataset_name, eval_cfg.dataset_path))
    from pathlib import Path
    cached_source = Path("initial_polyglot" if eval_cfg.polyglot else "initial_swe") / path_cfg.initial_agent_name / "src"
    if corrected and (path_cfg.root_cache_dir or path_cfg.initial_baseline_dir):
        cached_source = Path(__file__).resolve().parent
    proposed = build_manifest(config, output_dir, config.search.algorithm, list(dict.fromkeys(tasks)),
                              source=cached_source if cached_source.is_dir() else Path(__file__).resolve().parent,
                              require_clean=False, shuffle=not eval_cfg.polyglot,
                              seed=config.evaluation.task_order_seed if corrected else 42 if not eval_cfg.polyglot else None,
                              effective_branch_count=effective_branch_count)
    verified_initial = None
    if corrected:
        verified_initial, root_identity = prepare_search_root(config, output_dir, proposed)
        proposed["root_initialization"] = root_identity
        from run_metadata import fingerprint_inputs, digest
        proposed["fingerprint_inputs"] = fingerprint_inputs(proposed)
        proposed["reproducibility_fingerprint"] = digest(proposed["fingerprint_inputs"])
    identity = open_manifest(output_dir, proposed, resume=bool(path_cfg.continue_from))
    recorder = RunRecorder(output_dir, identity, initial_elapsed=time.monotonic() - run_started,
                           network_readiness=config.network_readiness)
    search_state = None
    checkpoint = None
    if corrected and path_cfg.continue_from:
        checkpoint = load_checkpoint(output_dir)
        if checkpoint["fingerprint"] != identity["reproducibility_fingerprint"]:
            raise ValueError("Search checkpoint and run manifest identities differ")
    diagnose = partial(self_improve_step.diagnose_problem, backend=llm_cfg.diagnose_backend,
                       codex_config=config.codex, legacy_model=llm_cfg.diagnose_llm,
                       log=hgm_utils.safe_log, downstream_backend=llm_cfg.downstream_backend,
                       recorder=recorder)
    if corrected:
        diagnose = partial(diagnose, context_config=config.diagnosis_context)
    try:
        if checkpoint is not None:
            saved = checkpoint["state"]
            pending = None
            if saved.get("inflight") and saved["inflight"]["kind"] == "expansion":
                if not saved["history"] or saved["history"][-1]["expansion_id"] != saved["inflight"]["expansion_id"]:
                    raise ProvenanceError("Pending candidate identity differs from search history", code="PROV_PENDING_ID")
                pending = saved["history"][-1]
            validate_search_nodes(output_dir, saved["nodes"], pending)
            hgm_utils.init(eval_cfg.polyglot, output_dir, tasks, 0, llm_cfg.self_improve_llm,
                           exec_cfg.self_improve_timeout)
            submitted_ids = defaultdict(set)
            src_path = str(Path(output_dir) / "initial/src")
            search_state = SearchState(config, hgm_utils.nodes, submitted_ids, hgm_utils.total_tasks)
            hgm_utils.n_task_evals = search_state.restore(checkpoint["state"], Node)
            hgm_utils.n_task_evals = search_state.reconcile(output_dir, Node, hgm_utils.n_task_evals)
        else:
            src_path, submitted_ids = initialize_run(
                output_dir,
                llm_cfg.self_improve_llm,
                tasks,
                path_cfg.initial_agent_name,
                prevrun_dir=path_cfg.continue_from,
                polyglot=eval_cfg.polyglot,
                timeout=exec_cfg.self_improve_timeout,
                max_workers=exec_cfg.max_workers,
                config=config,
                recorder=recorder,
                **({"verified_initial": verified_initial} if corrected else {}),
            )
            if corrected:
                search_state = SearchState(config, hgm_utils.nodes, submitted_ids, hgm_utils.total_tasks)
    except BaseException as error:
        recorder.refresh("interrupted" if isinstance(error, KeyboardInterrupt) else "failed")
        if run_lease is not None:
            run_lease.close()
        raise
    recorder.refresh()
    total_num_tasks = len(hgm_utils.total_tasks)

    def checkpoint_search():
        if search_state is not None:
            save_checkpoint(output_dir, search_state, hgm_utils.n_task_evals,
                            recorder.manifest["reproducibility_fingerprint"])

    checkpoint_search()

    # Set up logger
    logger.info(
        f"Starting HGM run {run_id} with configuration: {safe_configuration(config)}"
    )

    def TS_sample(evals):
        alphas = [1 + np.sum(de) for de in evals]
        betas = [1 + len(de) - np.sum(de) for de in evals]
        if opt_cfg.cool_down:
            alphas = np.array(alphas) * (
                10000
                if exec_cfg.max_task_evals == hgm_utils.n_task_evals
                else exec_cfg.max_task_evals**opt_cfg.beta
                / (exec_cfg.max_task_evals - hgm_utils.n_task_evals) ** opt_cfg.beta
            )
            betas = np.array(betas) * (
                10000
                if exec_cfg.max_task_evals == hgm_utils.n_task_evals
                else exec_cfg.max_task_evals**opt_cfg.beta
                / (exec_cfg.max_task_evals - hgm_utils.n_task_evals) ** opt_cfg.beta
            )
        thetas = np.random.beta(alphas, betas)
        return np.argmax(thetas)

    n_pending_expands = 0
    n_pending_measures = 0
    lock = threading.Lock()

    def expand():
        with lock:
            nodes = [
                node
                for node in hgm_utils.nodes.values()
                if np.isfinite(node.mean_utility) and node.mean_utility > 0
            ]
            candidate_options = {}
            if search_state is not None:
                if not nodes or not search_state.can_expand(exec_cfg.max_task_evals-hgm_utils.n_task_evals):
                    return
                selected_node, candidate_id = search_state.begin_expansion(
                    nodes, TS_sample, opt_cfg.n_pseudo_descendant_evals)
                candidate_options["candidate_id"] = candidate_id
                checkpoint_search()
            else:
                decendant_evals = [
                    node.get_decendant_evals(num_pseudo=opt_cfg.n_pseudo_descendant_evals)
                    for node in nodes
                ]
                selected_node = nodes[TS_sample(decendant_evals)]
        child_commit = hgm_utils.sample_child(
            selected_node.commit_id,
            image_name=path_cfg.initial_agent_name + ":latest",
            diagnoser=diagnose,
            self_improve_backend=llm_cfg.self_improve_backend,
            src_path=src_path,
            codex_config=config.codex,
            candidate_config=config.candidate,
            diagnosis_dataset=diagnosis_dataset,
            recorder=recorder,
            **candidate_options,
        )
        with lock:
            if child_commit != "failed":
                require_candidate(output_dir, child_commit, selected_node.commit_id)
                child = Node(child_commit, parent_id=selected_node.id)
                selected_node.children.append(child)
                if search_state is not None:
                    search_state.end_expansion(child)
                update_metadata(output_dir, hgm_utils.n_task_evals)
            elif search_state is not None:
                search_state.end_expansion(None)
            if search_state is not None:
                checkpoint_search()

    def sample():
        if search_state is not None and hgm_utils.n_task_evals >= exec_cfg.max_task_evals:
            return
        time.sleep(random.random())
        with lock:
            nonlocal n_pending_expands, n_pending_measures
            if hgm_utils.n_task_evals >= exec_cfg.max_task_evals:
                return

            if (
                hgm_utils.n_task_evals**opt_cfg.alpha
                >= len(hgm_utils.nodes) - 1 + n_pending_expands
            ):
                n_pending_expands += 1
                is_expand = True
            else:
                is_expand = False
            if search_state is not None and is_expand and (
                    not search_state.can_expand(exec_cfg.max_task_evals-hgm_utils.n_task_evals)
                    or not any(np.isfinite(n.mean_utility) and n.mean_utility > 0 for n in hgm_utils.nodes.values())):
                n_pending_expands -= 1
                is_expand = False
        if is_expand:
            expand()
            with lock:
                n_pending_expands -= 1
                return

        with lock:
            nodes = hgm_utils.nodes[0].get_sub_tree(fn=lambda node: node)
            nodes = [
                node for node in nodes if len(submitted_ids[node.id]) < total_num_tasks
            ]
            evals = [node.utility_measures for node in nodes]
            if len(evals) == 0:
                return
            selected_node = nodes[TS_sample(evals)]
            available_tasks = list(
                [
                    task
                    for task in hgm_utils.total_tasks
                    if task not in submitted_ids[selected_node.id]
                ]
            )
            if len(available_tasks) == 0:
                return
            if random.random() < opt_cfg.eval_random_level:
                selected_node_tasks = random.choice(available_tasks)
            else:
                selected_node_tasks = available_tasks[0]
            submitted_ids[selected_node.id].add(selected_node_tasks)
            n_pending_measures += 1
            if search_state is not None:
                search_state.begin_evaluation(selected_node, selected_node_tasks)
                checkpoint_search()

        evals = hgm_utils.eval_agent(
            selected_node.commit_id,
            tasks=[selected_node_tasks],
            init_agent_path=src_path,
            config=config,
            recorder=recorder,
        )
        with lock:
            if search_state is None:
                selected_node.utility_measures += evals
            else:
                if len(evals) != 1:
                    raise ValueError("Expected exactly one task result")
                search_state.observe(selected_node, selected_node_tasks, evals[0])
            n_pending_measures -= 1
            update_metadata(output_dir, hgm_utils.n_task_evals)
            if search_state is not None:
                checkpoint_search()

    run_status = "completed"
    try:
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = [
                executor.submit(cancellable_operation(config.codex, expand))
                for _ in range(
                    (search_state.initial_attempts if search_state is not None else len(hgm_utils.nodes) - 1),
                    (5 if search_state is not None and not search_state.initialized else
                     0 if search_state is not None else min(5, int(exec_cfg.max_workers**opt_cfg.alpha))),
                )
            ]
            try:
                for future in as_completed(futures):
                    future.result()
            except BaseException:
                for future in futures:
                    future.cancel()
                raise

        if search_state is not None and not search_state.initialized:
            search_state.freeze()
            finalize_search_manifest(recorder, search_state.identity())
            checkpoint_search()

        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = [
                executor.submit(cancellable_operation(config.codex, sample))
                for _ in range(int(exec_cfg.max_task_evals * 100))
            ]
            try:
                for future in as_completed(futures):
                    future.result()
            except BaseException:
                for future in futures:
                    future.cancel()
                raise

        if search_state is not None:
            if hgm_utils.n_task_evals != exec_cfg.max_task_evals:
                raise RuntimeError("Search ended before exhausting its evaluation budget")
            final = run_final_evaluation(config, output_dir, list(hgm_utils.nodes.values()), src_path, recorder)
            if final["status"] != "completed":
                raise RuntimeError("Final evaluation contains unsuccessful task executions")

    except ReadinessAbort:
        run_status = "incomplete"
        raise
    except Exception as e:
        run_status = "failed"
        logger.error(f"Error: {e}")
        logger.error(traceback.format_exc())
        print(repr(e))
    except KeyboardInterrupt:
        run_status = "incomplete" if readiness_cancel is not None else "interrupted"
        raise
    finally:
        if search_state is not None:
            # In-flight failures keep the pre-operation RNG snapshot on disk.
            if search_state.inflight is None:
                checkpoint_search()
        recorder.refresh(run_status)
        from pathlib import Path
        from candidate_workspace import _atomic_write
        _atomic_write(Path(output_dir) / "run_timing.json", json.dumps({
            "whole_run_total": time.monotonic() - run_started,
            "n_task_evals": hgm_utils.n_task_evals,
            "configured_workers": exec_cfg.max_workers, "effective_workers": effective_workers,
        }, indent=2).encode())
        if run_lease is not None:
            run_lease.close()
    if corrected and run_status == "failed":
        raise RuntimeError("Corrected search failed; see this run's search state and outer log")


if __name__ == "__main__":
    main()
