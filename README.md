# WGM: Lineage-Level Exploration for Self-Improving Coding Agents

This repository contains the implementation of **WGM**, a self-improving coding-agent
search that allocates expansions across a fixed set of lineages with EXP3 and selects
agents with best-arm identification (BAI), together with the HGM and DGM baselines
used for comparison. All three algorithms share one validated execution layer:
an append-only evaluation ledger, deterministic task scheduling, resumable search
operations and read-only reports.

> **Safety.** The code executes model-generated code (candidate agents and their
> patches). Run it only in an isolated environment.

## Algorithms

| Algorithm | Entry | Description |
| --- | --- | --- |
| **WGM** | `--algorithm wgm_v2` | Lineage-level EXP3 over the lineages founded during initialization, local BAI inside the selected lineage, founder/leader parent choice, and checkpoint BAI for the returned agent |
| HGM | `--algorithm hgm --hgm_profile legacy_code` | Clade-metaproductivity Thompson sampling baseline |
| DGM | `--algorithm dgm --dgm-profile legacy_code` | Archive-based open-ended search baseline |

A WGM round:

1. **Initialization.** The root is expanded a fixed number of times; each admitted
   child founds a lineage, and the lineage set is then frozen.
2. **Lineage selection.** A lineage is sampled with EXP3 (exploration rate `gamma`,
   learning rate `eta`).
3. **Local BAI.** A bounded number of evaluations (`local_bai_evals`) is spent on the
   leader/challenger inside the lineage, using anytime Hoeffding intervals
   (`delta`, stopping tolerance `epsilon`).
4. **Expansion.** The parent is the lineage founder with probability `rho_r` and the
   lineage leader otherwise; one child is generated, evaluated `M` times, and one
   importance-weighted EXP3 update is applied to the selected lineage.
5. **Checkpoint BAI.** At each configured checkpoint, a derived read-only BAI over the
   frozen archive (budget `checkpoint_bai_fraction` of the checkpoint budget) selects
   the agent that is evaluated on the fixed final task set.

`planning_horizon` is the fixed number of valid search evaluations; `target_budget`
is only a pause point, so a run can be continued checkpoint by checkpoint without
changing its identity. Ablations are expressed purely through configuration (for
example uniform lineage selection, or no local BAI).

## Agent backends

All model calls (diagnosis, self-improvement, downstream task solving) go through a
host agent CLI with account login; no API key is used.

* **Codex CLI** (default): `codex exec`, configured in the `codex:` section.
* **Claude Code CLI** (optional): set `codex.provider: claude_code`; see
  [CLAUDE_BACKEND.md](CLAUDE_BACKEND.md).

A run uses exactly one provider for every phase, and results of different providers
form separate conditions that are never mixed.

## Setup

```bash
conda create -n wgm python=3.10
conda activate wgm
python -m pip install -r requirements.txt   # legacy SDK entries are optional
```

Requirements on the host:

* Docker (grading runs in task containers).
* One agent CLI, installed and logged in (`codex`, with Node for the npm launcher,
  or `claude`).
* The SWE-bench harness, cloned into `swe_bench/SWE-bench` at the revision pinned in
  `pipeline.py` (`SWE_COMMIT`) and installed with `pip install -e swe_bench/SWE-bench`.
  Then fetch the dataset metadata with `python pipeline.py prepare-data`.
* Optional, for Polyglot: set `POLYGLOT_BENCHMARK_REPO` to the Polyglot benchmark
  git repository (or place a checkout at `polyglot/polyglot-benchmark`) and run
  `python -m polyglot.prepare_polyglot_dataset`.

Network locations are never hard-coded; provide them through the environment when
the corresponding feature is used:

| Variable | Used by |
| --- | --- |
| `POLYGLOT_BENCHMARK_REPO` | Polyglot dataset preparation (git clone source) |
| `WGM_GO_TARBALL_URL`, `WGM_RUSTUP_INIT_URL`, `WGM_NODESOURCE_SETUP_URL`, `WGM_MINICONDA_INSTALLER_URL` | Building the Polyglot base image (`$GOARCH` and `{conda_arch}` are substituted) |
| `WGM_NETWORK_PROBE_URL` | Printed resume commands of readiness-enabled runs |
| `OPENROUTER_BASE_URL` | Legacy API backend only |

Check the environment without any model call:

```bash
python pipeline.py preflight
```

Runs require a clean, committed worktree; every run records its commit and a
reproducibility fingerprint, and resume and final evaluation must use the same commit.

## Configuration

Runs are configured with a YAML file passed through `--config` plus command-line
overrides. The sections and their fields are defined in:

| Section | Definition |
| --- | --- |
| `llm`, `optimization`, `execution`, `evaluation`, `paths`, `search`, `network_readiness` | `config.py`, `search_config.py`, `network_readiness.py` |
| `codex` (agent CLI; `provider`, `model`, `reasoning_effort`, confinement) | `runners/codex_exec.py`, `runners/claude_code.py` |
| `wgm_v2` (WGM parameters, checkpoints, initial source) | `wgm_v2_config.py` |
| `dgm` | `dgm_legacy_config.py` |
| `hgm_incremental` | `hgm_incremental_config.py` |

Model and reasoning effort are always pinned explicitly (`--codex-model`,
`--reasoning-effort`, `--require-pinned-model`).

## Initial agent and verified root

All algorithms start from the same initial agent (this repository's agent source;
the evolvable downstream policy is `prompts/downstream_instructions.txt`). The root
is evaluated once on the benchmark tasks and sealed as a *verified root*:

```bash
CONFIG=my_config.yaml; MODEL=<model>; EFFORT=<effort>
mkdir -p results/root-source
git archive HEAD | tar -x -C results/root-source

python pipeline.py default-baseline --config "$CONFIG" --acknowledge-run --require-pinned-model \
  --codex-model "$MODEL" --reasoning-effort "$EFFORT" \
  --agent-path results/root-source --output-dir results/root-baseline
python pipeline.py import-root --config "$CONFIG" --codex-model "$MODEL" --reasoning-effort "$EFFORT" \
  --agent-path results/root-source \
  --baseline-dir results/root-baseline --root-cache-dir results/verified-root
```

WGM pre-registers the root identity: set `wgm_v2.initial_agent_source` to
`results/verified-root/initial/src` and `wgm_v2.initial_source_sha256` /
`wgm_v2.initial_policy_sha256` to the digests printed by

```bash
python -c "import pathlib, run_metadata as rm; s = pathlib.Path('results/verified-root/initial/src'); \
print(rm.source_snapshot(s)['sha256']); print(rm.file_hash(s / 'prompts/downstream_instructions.txt'))"
```

## Running the searches

```bash
COMMON=(--config "$CONFIG" --execution-policy validated --codex-model "$MODEL" --reasoning-effort "$EFFORT")

bash run.sh full-rsi --algorithm wgm_v2 "${COMMON[@]}" --target-budget <B> --output_dir results/wgm
bash run.sh full-rsi --algorithm hgm --hgm_profile legacy_code --root-cache-dir results/verified-root \
  "${COMMON[@]}" --target-budget <B> --output_dir results/hgm
bash run.sh full-rsi --algorithm dgm --dgm-profile legacy_code "${COMMON[@]}" --target-budget <B> \
  --output_dir results/dgm

# Continue a run to a later pause point with the same configuration
bash run.sh full-rsi --algorithm wgm_v2 --protocol formal "${COMMON[@]}" \
  --continue_from results/wgm --target-budget <B2>
```

Optionally add `--network-readiness --network-readiness-url <URL>` to wait for
connectivity before each call instead of failing. WGM and HGM searches are serial
(one adaptive decision at a time); independent fixed batches (root baseline, final
evaluations, DGM evaluation pool) accept `--evaluation-workers N`.

## Final evaluation and reports

```bash
# WGM: checkpoint BAI, then the final task set for the BAI-selected agent
python pipeline.py wgm-v2-bai        --output-dir results/wgm --checkpoint <B>
python pipeline.py wgm-v2-bai-final  --output-dir results/wgm --checkpoint <B>
python pipeline.py wgm-v2-bai-report --output-dir results/wgm --checkpoint <B>

# HGM / DGM: final evaluation of each method's returned agent
python pipeline.py final-eval --output-dir results/hgm --checkpoint <B> --root-cache-dir results/verified-root
python pipeline.py final-eval --output-dir results/dgm --checkpoint <B> --root-cache-dir results/verified-root

# Read-only comparison over complete finals, and the multi-checkpoint curve
python pipeline.py compare --checkpoint <B> --hgm-dir results/hgm --dgm-dir results/dgm \
  --wgm-dir results/wgm --output-dir results/report
python pipeline.py comparison-curve --hgm-dir results/hgm --dgm-dir results/dgm \
  --wgm-dir results/wgm --output-dir results/report-curve
```

`final-report`, `wgm-v2-report` and `report` are read-only. The canonical WGM
Primary final (`wgm-v2-final`, or `final-eval --roles primary-only`) is available as
an ablation of checkpoint BAI.

## Failure handling

Infrastructure failures are never recorded as task results. An interrupted or
failed call blocks the run until an operator authorizes exactly one retry with an
explicit, audited command, which writes an immutable grant and prints the resume
command:

* `final-audit --compact`, `final-authorize-preparation-retry`,
  `final-authorize-readiness-retry`, `final-cleanup-readiness`
* `search-authorize-diagnosis-retry`, `search-authorize-evaluation-retry`
* `dgm-audit`, `dgm-authorize-model-retry`, `dgm-cleanup`

## Licenses

This code builds on public Apache-2.0 implementations of the HGM and DGM baselines
and on the public SWE-bench and Polyglot evaluation harnesses; see `LICENSE`. Files derived from these projects have been modified.
