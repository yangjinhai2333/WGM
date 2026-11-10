# WGM: Lineage-Level Exploration for Self-Improving Coding Agents

WGM is a self-improving coding-agent search method that combines lineage-level
exploration with best-arm identification (BAI). This repository also includes
the HGM and DGM baselines.

## Setup

Use Python 3.10, Docker, and an installed, logged-in agent CLI.

```bash
conda create -n wgm python=3.10
conda activate wgm
python -m pip install -r requirements.txt
```

For SWE-bench, clone its harness into `swe_bench/SWE-bench` at the revision
specified by `SWE_COMMIT` in `pipeline.py`, then run:

```bash
pip install -e swe_bench/SWE-bench
python pipeline.py prepare-data
python pipeline.py preflight
```

Run model-generated code only in an isolated environment.

## Usage

Provide a YAML configuration, an explicit model and reasoning effort, and an
evaluation budget. Configuration fields are defined in `config.py` and
`wgm_v2_config.py`. WGM requires a verified root and its source digests in the
configuration. Runs require a clean, committed worktree.

```bash
bash run.sh full-rsi --algorithm wgm_v2 \
  --config <config.yaml> --execution-policy validated \
  --codex-model <model> --reasoning-effort <effort> \
  --target-budget <budget> --output_dir results/wgm
```

For available commands and options:

```bash
python pipeline.py --help
python hgm.py --help
```
