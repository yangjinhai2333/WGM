"""Explicit preflight, data preparation, one-task/baseline and timing commands.

No benchmark or model call occurs at import time. Preflight is the default
operation; larger runs require an explicit baseline task count or hgm.py.
"""

import argparse
from dataclasses import asdict, replace
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime

from candidate_workspace import CandidateConfig, _atomic_write, _git, generate_candidate
from config import HGMConfig, load_config
from downstream_solver import load_agent_policy, load_swe_dataset, select_entries
from runners import CodexExecRunner
from runners.codex_exec import _clean_environment, _stderr_summary
from run_metadata import build_manifest, open_manifest, RunRecorder, read_report, report_text


ROOT = Path(__file__).resolve().parent
SWE_COMMIT = "dc4c087c2b9e4cefebf2e3d201d27e362d899e0f"


def preflight(config, *, benchmark=True, runner=None, min_free_gib=20, output_root=None):
    """Read-only checks; no model, image build, login mutation or data download."""
    checks = []
    def check(name, ok, detail, fix=""):
        checks.append(dict(name=name, ok=bool(ok), detail=detail, fix=fix if not ok else ""))
    check("python", sys.version_info[:2] == (3, 10),
          f"{sys.executable} ({sys.version.split()[0]})", "conda activate wgm")
    check("conda", os.environ.get("CONDA_DEFAULT_ENV") == "wgm",
          "wgm environment required", "conda activate wgm")
    for executable in ("git", "node"):
        path = shutil.which(executable)
        good = False
        version = "missing"
        if path:
            try:
                result = subprocess.run([path, "--version"], capture_output=True, text=True,
                                        timeout=10, env=_clean_environment())
                good = result.returncode == 0
                version = result.stdout.strip() if good else "version check failed"
            except (OSError, subprocess.TimeoutExpired):
                version = "version check failed"
        check(executable, good, f"{path}: {version}", f"Ensure {executable} is on PATH after activating wgm")
    login = (runner or CodexExecRunner(config.codex)).preflight()
    if getattr(config.codex, "provider", None) == "claude_code":
        check("claude", login.ok, json.dumps(asdict(login)),
              "Check claude on PATH; run claude auth login, then claude auth status")
        if config.codex.confinement == "bwrap":
            check("bwrap", shutil.which(config.codex.bwrap_executable) is not None,
                  "bubblewrap confinement for Claude tool execution", "Install bubblewrap or configure bwrap_executable")
    else:
        check("codex", login.ok, json.dumps(asdict(login)),
              "Check codex/node PATH; run codex login --device-auth, then codex login status")
    target = Path(output_root or ROOT).resolve()
    while not target.exists():
        target = target.parent
    free = shutil.disk_usage(target).free / 2**30
    check("disk", free >= min_free_gib, f"{free:.1f} GiB free; minimum {min_free_gib} GiB",
          "Select a filesystem with more free space; do not globally prune Docker")
    if benchmark:
        for module in ("numpy", "yaml", "docker", "datasets", "pathspec", "swebench"):
            check(module, importlib.util.find_spec(module) is not None, f"Python module {module}",
                  "Install declared runtime dependencies; install the pinned SWE-bench checkout with python -m pip install --no-deps -e swe_bench/SWE-bench")
        try:
            result = subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"],
                                    capture_output=True, timeout=20, env=_clean_environment())
            check("docker_daemon", result.returncode == 0, "Docker daemon reachability",
                  "Check docker info and permission to access the configured Docker socket")
        except (OSError, subprocess.TimeoutExpired):
            check("docker_daemon", False, "Docker unavailable",
                  "Ensure Docker is installed, running and accessible to this user")
        # The CLI can work while this Python environment's SDK cannot connect.
        try:
            import docker
            client = docker.from_env(timeout=10)
            try:
                reachable = client.ping()
            finally:
                client.close()
            check("docker_sdk", reachable, "WGM Python Docker client ping",
                  "Check the WGM Docker SDK and Docker socket configuration")
        except Exception as error:
            check("docker_sdk", False, f"{type(error).__name__}: {_stderr_summary(str(error), '')}",
                  "Check Docker connectivity from WGM Python; CLI success alone is insufficient")
        checkout = ROOT / "swe_bench/SWE-bench"
        try:
            revision = _git(checkout, "rev-parse", "HEAD").decode().strip()
        except Exception:
            revision = "missing"
        check("swe_checkout", revision == SWE_COMMIT, revision,
              f"Prepare swe_bench/SWE-bench at {SWE_COMMIT}; see README")
        try:
            if config.evaluation.polyglot:
                path = Path(config.evaluation.polyglot_dataset_path)
                data = json.loads(path.read_text())
                ids = json.loads((ROOT / "polyglot/subsets/small.json").read_text())
                ids += json.loads((ROOT / "polyglot/subsets/medium.json").read_text())
                select_entries(data, ids)
                for entry in data:
                    if not (ROOT / entry["repo"]).is_dir():
                        raise ValueError("Polyglot task checkout missing")
            else:
                if (config.evaluation.full_eval and
                        config.evaluation.dataset_name != "princeton-nlp/SWE-bench_Verified"):
                    raise ValueError("Full evaluation requires the original Verified selection")
                data = load_swe_dataset(config.evaluation.dataset_name, config.evaluation.dataset_path)
                ids = json.loads((ROOT / "swe_bench/subsets/small.json").read_text())
                ids += json.loads((ROOT / "swe_bench/subsets/medium.json").read_text())
                select_entries(data, ids)
            check("dataset", True, f"{len(data)} entries, subset IDs validated")
        except Exception:
            fix = ("python -m polyglot.prepare_polyglot_dataset" if config.evaluation.polyglot
                   else "python pipeline.py prepare-data --config config.yaml; for full_eval use SWE-bench_Verified and a separate dataset_path")
            check("dataset", False, "Dataset/checkouts missing, mismatched, or subset IDs invalid", fix)
        check("agent_policy", (ROOT / "prompts/downstream_instructions.txt").is_file(),
              "Default evolvable downstream instructions", "Restore the declared runtime instructions")
    return {"ok": all(item["ok"] for item in checks), "checks": checks}


def require_preflight(config, **kwargs):
    result = preflight(config, **kwargs)
    for check in result["checks"]:
        print(f"{'OK' if check['ok'] else 'FAIL'} {check['name']}: {check['detail']}", flush=True)
        if check["fix"]:
            print(f"  Fix: {check['fix']}", flush=True)
    if not result["ok"]:
        raise RuntimeError("Preflight failed; no model or benchmark started")
    return result


def prepare_data(config):
    """Explicit metadata download, never an evaluation or image build."""
    from datasets import load_dataset
    destination = Path(config.evaluation.dataset_path).resolve()
    if destination.exists():
        raise ValueError("Dataset destination already exists; select a new path, no files were overwritten")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".wgm-dataset-", dir=destination.parent) as temporary:
        staged = Path(temporary) / "dataset"
        dataset = load_dataset(config.evaluation.dataset_name, split="test")
        dataset.save_to_disk(str(staged))
        (staged / "wgm_dataset.json").write_text(json.dumps({
            "dataset_name": config.evaluation.dataset_name, "split": "test",
            "rows": len(dataset), "fingerprint": dataset._fingerprint,
        }, indent=2))
        staged.rename(destination)
    print(f"Prepared {len(dataset)} metadata rows in {destination}; no evaluations.")


def run_baseline(config, out_dir, *, n_tasks=1, instance_ids=None, agent_path=None, resume=False,
                 run_type="single-task", require_pinned_model=False, command=None):
    """Fixed task list; validated batches may overlap independent task lifecycles."""
    from execution_policy import validated, check_resume
    if resume:
        check_resume(out_dir, config)
    if validated(config):
        from validated_runs import run_baseline as run_validated_baseline
        return run_validated_baseline(config, out_dir, n_tasks=n_tasks, instance_ids=instance_ids,
                                      agent_path=agent_path, resume=resume, run_type=run_type,
                                      require_pinned_model=require_pinned_model, command=command)
    if config.execution.evaluation_workers != 1:
        import warnings
        warnings.warn('evaluation_workers applies to validated fixed batches; origin retains its existing execution path', stacklevel=2)
    from swe_bench.harness import harness as swe_harness
    from polyglot.harness import harness as polyglot_harness
    from swe_bench.report import make_report
    started = time.monotonic()
    out_dir = Path(out_dir).resolve()
    manifest_path = out_dir / "run.json"
    agent_path = str(Path(agent_path or ROOT).resolve())
    from task_schedule import canonical_tasks, unique_tasks
    canonical = instance_ids is None
    if instance_ids is None:
        folder = ROOT / ("polyglot" if config.evaluation.polyglot else "swe_bench") / "subsets"
        instance_ids = json.loads((folder / "small.json").read_text())
        instance_ids += json.loads((folder / "medium.json").read_text())
    if canonical:
        # Both benchmarks use their canonical shuffled pool (Polyglot: 59 distinct tasks).
        instance_ids = canonical_tasks(ROOT, config.evaluation.task_order_seed, polyglot=config.evaluation.polyglot)
    ids = (list(dict.fromkeys(instance_ids)) if config.evaluation.polyglot else unique_tasks(instance_ids))[:n_tasks]
    if n_tasks < 1 or len(ids) != n_tasks:
        raise ValueError("Task count is invalid or exceeds the selected subset")
    proposed = build_manifest(config, out_dir, run_type, ids, source=agent_path,
                              require_pinned_model=require_pinned_model, command=command,
                              shuffle=canonical,
                              seed=config.evaluation.task_order_seed)
    if not resume:
        out_dir.mkdir(parents=True, exist_ok=False)
    identity = open_manifest(out_dir, proposed, resume=resume)
    if run_type == "default-baseline" and (out_dir / "baseline_seal.json").exists():
        from root_cache import verify_baseline_seal
        verify_baseline_seal(out_dir)
        return out_dir  # A completed sealed baseline is immutable, including on resume.
    if run_type == "default-baseline" and "initial_source_snapshot" in identity["policy"]:
        from run_metadata import copy_source_snapshot, source_snapshot
        snapshot = out_dir / "src"
        expected = identity["policy"]["initial_source_snapshot"]
        if snapshot.exists():
            if source_snapshot(snapshot) != expected:
                raise ValueError("Saved baseline source snapshot changed")
        else:
            copy_source_snapshot(agent_path, snapshot, expected)
    runtime_agent_path = str(out_dir / "src") if run_type == "default-baseline" and (out_dir / "src").is_dir() else agent_path
    print(f"Run {run_type}: tasks={len(ids)}, task_hash={identity['tasks']['sha256']}, "
          f"commit={identity['repository']['commit']}, fingerprint={identity['reproducibility_fingerprint']}, "
          f"effective_workers={identity['configuration']['effective']['effective']['workers']}, "
          f"model_pinned={bool(config.codex.model)}", flush=True)
    baseline_config = config.to_dict()
    baseline_config.pop("network_readiness", None)
    baseline_config.pop("search", None)
    baseline_config.pop("wgm", None)
    baseline_config.pop("wgm_v2", None)
    baseline_config.pop("protocol", None)
    baseline_config.pop("diagnosis_context", None)
    manifest = {"backend": config.llm.downstream_backend, "task_ids": ids,
                "agent_path": agent_path, "config": baseline_config, "status": "running",
                "started_at": datetime.now().isoformat()}
    previous_total = 0.0
    if config.llm.downstream_backend == "codex_exec":
        manifest["policy_sha256"] = hashlib.sha256(load_agent_policy(runtime_agent_path).encode()).hexdigest()
    if resume:
        previous = json.loads(manifest_path.read_text())
        if (previous["task_ids"] != ids or previous["config"] != manifest["config"]
                or previous["agent_path"] != agent_path
                or previous.get("policy_sha256") != manifest.get("policy_sha256")):
            raise ValueError("Resume arguments/config differ from the saved run")
        previous_total = previous.get("whole_run_total", 0.0)
        manifest["started_at"] = previous["started_at"]
    _atomic_write(manifest_path, json.dumps(manifest, indent=2).encode())
    recorder = RunRecorder(out_dir, identity, initial_elapsed=time.monotonic() - started,
                           network_readiness=config.network_readiness)
    status = "completed"
    predictions = out_dir / "predictions"
    kwargs = dict(
        test_task_list=ids, max_workers=config.execution.max_workers,
        model_name_or_path="baseline", init_agent_path=runtime_agent_path,
        pred_dname=str(out_dir), resume_dir=predictions, n_tasks=n_tasks,
        backend=config.llm.downstream_backend, codex_config=config.codex,
        evaluation_timeout=config.execution.evaluation_timeout,
        legacy_model=config.llm.downstream_llm,
        recorder=recorder,
    )
    try:
        if config.evaluation.polyglot:
            paths = polyglot_harness(dataset_path=config.evaluation.polyglot_dataset_path,
                                    output_dir=out_dir, **kwargs)
        else:
            paths = swe_harness(dataset_name=config.evaluation.dataset_name,
                                dataset_path=config.evaluation.dataset_path, **kwargs)
            make_report(paths, output_dir=out_dir, num_eval_procs=1,
                        dataset_name=config.evaluation.dataset_name)
        if config.llm.downstream_backend == "codex_exec":
            report = json.loads((predictions / "batch.json").read_text())["report"]
            failed = sum(report[key + "_instances"] for key in ("incomplete", "error", "empty_patch"))
            if failed:
                raise RuntimeError(
                    f"{failed} task(s) failed to execute; "
                    f"see {predictions / 'tasks'} for metadata.json and error.log"
                )
        manifest["status"] = "complete"
    except BaseException as error:
        from network_readiness import ReadinessAbort
        status = ("incomplete" if isinstance(error, ReadinessAbort) else
                  "interrupted" if isinstance(error, KeyboardInterrupt) else "failed")
        manifest["status"] = "interrupted_or_failed"
        raise
    finally:
        manifest["whole_run_total"] = previous_total + time.monotonic() - started
        _atomic_write(manifest_path, json.dumps(manifest, indent=2).encode())
        recorder.refresh(status)
    if run_type == "default-baseline" and status == "completed":
        from root_cache import seal_baseline
        seal_baseline(out_dir)
    return out_dir


def estimate(directory, max_task_evals=800):
    """Observed cold preparation plus median warm task time, explicitly bounded."""
    batches = [json.loads(path.read_text()) for path in Path(directory).rglob("batch.json")]
    tasks = [task for batch in batches for task in batch["timing"]["tasks"]]
    if not tasks:
        raise ValueError("No task timing records found")
    cold = sum(b["timing"].get("dataset_preparation", 0) for b in batches)
    cold += sum(t["dataset_image_preparation"] for t in tasks if t.get("instance_image_cached") is not True)
    hot = statistics.median(
        t["per_task_total"] - (t["dataset_image_preparation"] if t.get("instance_image_cached") is not True else 0)
        for t in tasks)
    budget = max(t["timeout_seconds"] for t in tasks)
    return {
        "units": "seconds", "samples": len(tasks), "cold_setup_time": cold,
        "median_observed_warm_task_time": hot,
        "estimated_10_task_time": cold + 10 * hot,
        "estimated_60_task_time": cold + 60 * hot,
        "estimated_max_task_evals_time": cold + max_task_evals * hot,
        "max_task_evals": max_task_evals,
        "timeout_based_model_and_eval_upper_bound": max_task_evals * 2 * budget,
        "assumptions": "Serial execution and reused images. Different repos may require additional cold builds. "
                       "The timeout bound covers model+evaluation only; builds, archive I/O and reporting have no global wall-clock bound. "
                       "RSI diagnosis/self-improvement calls are additional, not included.",
    }


def candidate_smoke(config, output):
    """One real Codex call in a synthetic source; never imports or calls Docker."""
    from candidate_provenance import require_candidate_backend
    require_candidate_backend(config.llm.self_improve_backend)
    with tempfile.TemporaryDirectory(prefix="wgm-candidate-smoke-") as directory:
        source = Path(directory) / "source"
        source.mkdir()
        (source / "calc.py").write_text("def add(a, b):\n    return a - b\n")
        (source / "test_calc.py").write_text(
            "import unittest\nfrom calc import add\nclass TestAdd(unittest.TestCase):\n"
            "    def test_add(self):\n        self.assertEqual(add(2, 3), 5)\n")
        _git(source, "init", "-q", "--template=")
        _git(source, "add", ".")
        _git(source, "-c", "user.name=Smoke", "-c", "user.email=smoke@localhost",
             "commit", "-qm", "Synthetic source")
        before = _git(source, "status", "--porcelain")
        result = generate_candidate(
            source, Path(output) / "candidate", [],
            "Fix calc.add to return a + b. Keep test_calc.py unchanged.",
            "Run python -m unittest -v test_calc. No Docker, benchmarks or further model calls.",
            codex_config=replace(config.codex, timeout_seconds=min(180, config.codex.timeout_seconds)),
            config=CandidateConfig(validation_command=[sys.executable, "-m", "unittest", "-v", "test_calc"]),
        )
        if not result.ok:
            raise RuntimeError(f"Candidate smoke failed: {result.metadata.get('stage')}: {result.metadata.get('reason')}")
        assert result.patch_path.stat().st_size > 0
        _git(source, "apply", "--check", input=result.patch_path.read_bytes())
        assert _git(source, "status", "--porcelain") == before
        assert (source / "calc.py").read_text() == "def add(a, b):\n    return a - b\n"
    _atomic_write(Path(output) / "smoke.json", b'{"status":"passed","codex_calls":1,"docker_calls":0}\n')
    print("Host candidate smoke passed")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    from network_readiness import add_cli, configure_cli
    add_cli(parser)
    parser.add_argument("mode", nargs="?", default="preflight",
                        choices=("preflight", "prepare-data", "smoke", "single-task",
                                 "small-baseline", "default-baseline", "estimate", "report", "import-root",
                                 "wgm-v2-bai", "wgm-v2-final", "wgm-v2-report", "wgm-v2-bai-final", "wgm-v2-bai-report",
                                 "final-eval", "final-report", "final-audit", "final-authorize-preparation-retry", "compare",
                                 "final-authorize-readiness-retry", "final-cleanup-readiness", "comparison-curve",
                                 "dgm-audit", "dgm-authorize-model-retry", "dgm-cleanup",
                                 "search-authorize-diagnosis-retry", "search-authorize-evaluation-retry"))
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--execution-policy", choices=("origin", "validated"), default=None,
                        help="Override execution.policy in config; defaults to origin when neither is set")
    parser.add_argument("--output-dir")
    parser.add_argument("--timing-dir")
    parser.add_argument("--checkpoint", type=int)
    parser.add_argument('--wgm-result', choices=('bai', 'canonical'), default='bai',
                        help='Comparison result: BAI-selected final (default), or explicit canonical ablation')
    parser.add_argument('--retry-preparation', help='Use an explicit preparation authorization token (legacy terminal SHA256 also accepted)')
    parser.add_argument('--run-log', help='Search run log holding the Claude CLI result error (evaluation retry)')
    parser.add_argument('--evaluation-id', help='Evaluation to audit and authorize for one preparation retry')
    parser.add_argument('--compact', action='store_true', help='Compact, read-only final or DGM audit')
    parser.add_argument('--acknowledge-unknown-outcome', action='store_true',
                        help='Explicitly abandon an unconfirmed DGM model attempt; outcome remains unknown')
    parser.add_argument("--hgm-dir")
    parser.add_argument("--dgm-dir")
    parser.add_argument("--wgm-dir")
    parser.add_argument("--roles", choices=("primary-only", "primary-secondary", "secondary-only", "both"), default="primary-only")
    parser.add_argument("--evaluation-workers", type=int, default=None,
                        help="Independent validated fixed-task evaluations (default 1); adaptive search stays serial")
    parser.add_argument("--agent-path", default=str(ROOT))
    parser.add_argument("--instance-id", action="append")
    parser.add_argument("--n-tasks", type=int)
    parser.add_argument("--max-task-evals", type=int, default=800)
    parser.add_argument("--polyglot", action="store_true", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--codex-model")
    parser.add_argument("--reasoning-effort", choices=("minimal", "low", "medium", "high", "xhigh", "max"))
    parser.add_argument("--require-pinned-model", action="store_true")
    parser.add_argument("--task-order-seed", type=int)
    parser.add_argument("--baseline-dir")
    parser.add_argument("--root-cache-dir")
    parser.add_argument("--acknowledge-run", action="store_true")
    parser.add_argument("--json", action="store_true", help="Machine-readable report, no ordinary logs")
    args = parser.parse_args(argv)
    if args.mode == 'comparison-curve':
        if not all((args.hgm_dir, args.dgm_dir, args.wgm_dir, args.output_dir)):
            parser.error('comparison-curve requires --hgm-dir, --dgm-dir, --wgm-dir and --output-dir')
        from comparison_curve import curve, text_report
        result = curve({'hgm':args.hgm_dir, 'dgm':args.dgm_dir, 'wgm_v2':args.wgm_dir}, args.output_dir,
                       wgm_result=args.wgm_result)
        print(json.dumps(result, indent=2, sort_keys=True) if args.json else text_report(result))
        return
    if args.mode in ('final-authorize-readiness-retry', 'final-cleanup-readiness'):
        if not args.output_dir or args.checkpoint is None or args.retry_preparation or args.acknowledge_unknown_outcome:
            parser.error('readiness recovery requires --output-dir and --checkpoint, without other retry flags')
        from final_readiness_recovery import authorize_final, cleanup
        if args.mode == 'final-authorize-readiness-retry':
            if not args.evaluation_id:
                parser.error('readiness authorization requires --evaluation-id')
            result = authorize_final(args.output_dir, args.checkpoint, args.evaluation_id)
        else:
            if args.evaluation_id:
                parser.error('cleanup audits ownership of every pending resource in this final')
            result = cleanup(args.output_dir, args.checkpoint)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.mode == 'search-authorize-diagnosis-retry':
        if not args.output_dir:
            parser.error('search-authorize-diagnosis-retry requires --output-dir')
        from search_operation_retry import authorize_diagnosis_retry
        print(json.dumps(authorize_diagnosis_retry(args.output_dir, acknowledge=args.acknowledge_unknown_outcome),
                         indent=2, sort_keys=True))
        return
    if args.mode == 'search-authorize-evaluation-retry':
        if not args.output_dir or not args.run_log:
            parser.error('search-authorize-evaluation-retry requires --output-dir and --run-log')
        from evaluation_retry import authorize_evaluation_retry
        print(json.dumps(authorize_evaluation_retry(args.output_dir, args.run_log,
                                                    acknowledge=args.acknowledge_unknown_outcome),
                         indent=2, sort_keys=True))
        return
    if args.mode in ('dgm-audit', 'dgm-authorize-model-retry', 'dgm-cleanup'):
        if not args.output_dir:
            parser.error('DGM recovery commands require --output-dir')
        if args.retry_preparation or args.mode != 'dgm-authorize-model-retry' and (
                args.evaluation_id or args.acknowledge_unknown_outcome):
            parser.error('recovery authorization flags do not apply to this command')
        if args.compact and args.mode != 'dgm-audit':
            parser.error('--compact is only supported by read-only audits')
        from dgm_recovery import audit, authorize, cleanup
        if args.mode == 'dgm-authorize-model-retry':
            if not args.evaluation_id or not args.acknowledge_unknown_outcome:
                parser.error('authorization requires --evaluation-id and --acknowledge-unknown-outcome')
            result = authorize(args.output_dir, args.evaluation_id, acknowledge=True)
        elif args.mode == 'dgm-audit':
            result = audit(args.output_dir, compact=args.compact)
        else:
            result = cleanup(args.output_dir)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.acknowledge_unknown_outcome:
        parser.error('--acknowledge-unknown-outcome is only supported by dgm-authorize-model-retry')
    if args.retry_preparation and args.mode not in ('final-eval', 'wgm-v2-bai-final'):
        parser.error('--retry-preparation requires a final execution command')
    if args.compact and args.mode != 'final-audit':
        parser.error('--compact is only supported by final-audit or dgm-audit')
    if args.evaluation_id and args.mode != 'final-authorize-preparation-retry':
        parser.error('--evaluation-id is only supported by final-authorize-preparation-retry')
    if args.mode == 'final-authorize-preparation-retry':
        if not args.output_dir or args.checkpoint is None or not args.evaluation_id:
            parser.error('authorization requires --output-dir, --checkpoint and --evaluation-id')
        from preparation_authorization import authorize_final
        result = authorize_final(args.output_dir, args.checkpoint, args.evaluation_id)
        print(json.dumps(result, indent=2, sort_keys=True) if args.json else
              f"Authorization token: {result['token']}\nModel calls: 0; evaluator calls: 0\n{result['recommended_command']}")
        return
    if args.mode == 'final-audit':
        if not args.output_dir or args.checkpoint is None:
            parser.error('final-audit requires --output-dir and --checkpoint')
        if args.roles in ('both', 'primary-secondary'):
            parser.error('final-audit accepts one role at a time')
        from final_preparation_audit import audit, compact_audit
        audit_fn = compact_audit if args.compact else audit
        result = audit_fn(args.output_dir, args.checkpoint,
                          **({'role': 'secondary'} if args.roles == 'secondary-only' else {}))
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.mode == 'compare':
        if not all((args.hgm_dir, args.dgm_dir, args.wgm_dir, args.output_dir, args.checkpoint)):
            parser.error('compare requires --hgm-dir, --dgm-dir, --wgm-dir, --output-dir and --checkpoint')
        from comparison_report import compare, report_text as comparison_text
        result = compare({'hgm': args.hgm_dir, 'dgm': args.dgm_dir, 'wgm_v2': args.wgm_dir},
                         args.checkpoint, args.output_dir, wgm_result=args.wgm_result)
        print(json.dumps(result, indent=2, sort_keys=True) if args.json else comparison_text(result))
        return
    if args.mode in ('wgm-v2-bai', 'wgm-v2-bai-final', 'wgm-v2-bai-report'):
        if not args.output_dir or args.checkpoint is None:
            parser.error('checkpoint BAI commands require --output-dir and --checkpoint')
        from wgm_bai_final import run as run_bai, report_bundle
        from final_adapter import config_from_manifest
        manifest = json.loads((Path(args.output_dir) / 'run_manifest.json').read_text())
        saved = config_from_manifest(manifest)
        if args.roles != 'primary-only':
            parser.error('checkpoint BAI has one frozen winner; roles do not apply')
        if args.config != str(ROOT / 'config.yaml') or args.root_cache_dir is not None or (
                args.task_order_seed is not None and args.task_order_seed != manifest['tasks']['task_order_seed']):
            parser.error('checkpoint BAI restores the frozen manifest; omit config/root/schedule overrides')
        if ((args.codex_model is not None and args.codex_model != saved.codex.model) or
            (args.reasoning_effort is not None and args.reasoning_effort != saved.codex.reasoning_effort) or
            args.execution_policy not in (None, 'validated')):
            parser.error('checkpoint BAI uses the frozen manifest model/reasoning/policy')
        configure_cli(saved, args)
        result = (report_bundle(args.output_dir, args.checkpoint) if args.mode == 'wgm-v2-bai-report' else
            run_bai(args.output_dir, args.checkpoint, 'bai' if args.mode == 'wgm-v2-bai' else 'final',
                    workers=args.evaluation_workers, preflight=require_preflight,
                    retry_preparation=args.retry_preparation, readiness=saved.network_readiness))
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.mode in ('final-eval', 'final-report'):
        if not args.output_dir or args.checkpoint is None:
            parser.error('final adapter requires --output-dir and --checkpoint')
        from final_adapter import config_from_manifest, run_final, final_report, final_text
        if args.mode == 'final-report':
            result = final_report(args.output_dir, args.checkpoint)
        else:
            manifest = json.loads((Path(args.output_dir) / 'run_manifest.json').read_text())
            config = config_from_manifest(manifest, args.evaluation_workers)
            if ((args.codex_model is not None and args.codex_model != config.codex.model) or
                (args.reasoning_effort is not None and args.reasoning_effort != config.codex.reasoning_effort) or
                (args.execution_policy is not None and args.execution_policy != 'validated')):
                parser.error('final execution must use the saved model/reasoning and validated policy')
            configure_cli(config, args)
            recovery = {'retry_preparation': args.retry_preparation} if args.retry_preparation else {}
            result = run_final(args.output_dir, args.checkpoint, workers=args.evaluation_workers,
                roles=args.roles, root_cache=args.root_cache_dir, readiness=config.network_readiness,
                preflight=require_preflight, **recovery)
        print(json.dumps(result, indent=2, sort_keys=True) if args.json else final_text(result))
        return
    if args.mode == "wgm-v2-report":
        if not args.output_dir or args.checkpoint is None:
            parser.error("wgm-v2-report requires --output-dir and --checkpoint")
        from wgm_v2_derived import Checkpoint, report
        print(json.dumps(report(Checkpoint(args.output_dir, args.checkpoint)), indent=2, sort_keys=True))
        return
    if args.mode == "report":
        if not args.output_dir:
            parser.error("report requires --output-dir")
        try:
            identity, summary, old_run = read_report(args.output_dir)
        except (ValueError, KeyError, OSError) as error:
            parser.exit(1, f"Report failed: {error}\n")
        print(json.dumps(summary, sort_keys=True) if args.json else report_text(identity, summary, old_run))
        return
    if args.mode in ("small-baseline", "default-baseline") and not args.acknowledge_run:
        parser.error("Baseline execution requires --acknowledge-run (non-interactive confirmation)")
    config = load_config(args.config, **{"evaluation.polyglot": args.polyglot,
                         "execution.evaluation_workers": args.evaluation_workers,
                         "codex.model": args.codex_model, "codex.reasoning_effort": args.reasoning_effort,
                         "evaluation.task_order_seed": args.task_order_seed})
    from execution_policy import configure, check_resume
    configure(config, args.execution_policy)
    configure_cli(config, args)
    if args.resume and args.output_dir:
        check_resume(args.output_dir, config)
    if args.require_pinned_model and not config.codex.model:
        parser.error("--require-pinned-model requires codex.model or --codex-model")
    for attr in ("dataset_path", "polyglot_dataset_path"):
        path = Path(getattr(config.evaluation, attr))
        if not path.is_absolute():
            setattr(config.evaluation, attr, str(ROOT / path))
    if args.mode in ("wgm-v2-bai", "wgm-v2-final"):
        if not args.output_dir or args.checkpoint is None:
            parser.error("V2 derived evaluation requires --output-dir and --checkpoint")
        from wgm_v2_derived import run_derived
        result = run_derived(config, args.output_dir, args.checkpoint,
            "bai" if args.mode == "wgm-v2-bai" else "final", roles=args.roles,
            root_cache=args.root_cache_dir, preflight=require_preflight)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.mode == "import-root":
        if not args.baseline_dir or not args.root_cache_dir:
            parser.error("import-root requires --baseline-dir and --root-cache-dir")
        from root_cache import import_baseline, root_reference
        from task_schedule import canonical_tasks
        baseline = json.loads((Path(args.baseline_dir) / "run_manifest.json").read_text())
        # Import is offline: reuse recorded host probe facts, recompute source,
        # dataset and evaluator identities locally, and never call preflight.
        proposed = build_manifest(config, args.root_cache_dir, "default-baseline",
            canonical_tasks(ROOT, config.evaluation.task_order_seed, polyglot=config.evaluation.polyglot),
            source=args.agent_path,
            environment=baseline["environment"], shuffle=True, seed=config.evaluation.task_order_seed)
        print(json.dumps(root_reference(import_baseline(args.baseline_dir, args.root_cache_dir, proposed)), sort_keys=True))
        return
    if args.mode == "estimate":
        if not args.timing_dir:
            parser.error("--timing-dir is required")
        report = estimate(args.timing_dir, args.max_task_evals)
        output = Path(args.output_dir) if args.output_dir else ROOT / "results" / (
            "estimate-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
        output.mkdir(parents=True, exist_ok=False)
        _atomic_write(output / "estimate.json", json.dumps(report, indent=2).encode())
        print(json.dumps(report, indent=2))
        print(f"Results: {output}")
        return
    if args.mode == "prepare-data":
        prepare_data(config)
        return
    output = Path(args.output_dir) if args.output_dir else ROOT / "results" / (
        args.mode + "-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    if args.mode == "preflight":
        output.mkdir(parents=True, exist_ok=False)
        report = preflight(config, output_root=output)
        _atomic_write(output / "preflight.json", json.dumps(report, indent=2).encode())
        print(json.dumps(report, indent=2))
        if not report["ok"]:
            raise SystemExit(1)
        return
    require_preflight(config, benchmark=args.mode != "smoke", output_root=output)
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    if args.mode == "smoke":
        output.mkdir(parents=True, exist_ok=False)
        candidate_smoke(config, output)
    else:
        count = args.n_tasks if args.n_tasks is not None else {
            "single-task": 1, "small-baseline": 10, "default-baseline": 60}[args.mode]
        if args.mode == "single-task" and count != 1:
            parser.error("single-task permits exactly one task")
        if args.mode == "default-baseline" and config.evaluation.polyglot and args.n_tasks is None:
            # Preserve HGM's existing deduplication: Polyglot subsets overlap.
            small = json.loads((ROOT / "polyglot/subsets/small.json").read_text())
            medium = json.loads((ROOT / "polyglot/subsets/medium.json").read_text())
            count = len(set(small + medium))
            print(f"Polyglot small+medium contains {count} distinct tasks; each is solved once.")
        run_baseline(config, output, n_tasks=count, instance_ids=args.instance_id,
                     agent_path=args.agent_path, resume=args.resume, run_type=args.mode,
                     require_pinned_model=args.require_pinned_model,
                     command=["pipeline.py", *(argv if argv is not None else sys.argv[1:])])
    print(f"Results: {output}", flush=True)


if __name__ == "__main__":
    main()
