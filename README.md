# Wright–Gödel Machine (WGM): Mitigating Homogenization in Self-Improving Agents

WGM is an archive-based self-improving coding-agent search method. It selects
parents with an EXP3-based rule that combines clade statistics with a uniform
exploration floor, and uses best-arm identification (BAI) to allocate
additional evaluations and select the final agent (Algorithm 1 of the paper).
The implementation is in `wgm_v2_selector.py`, `wgm_v2_state.py` and
`wgm_v2_search.py`; the final-selection stage is in `wgm_v2_derived.py`.
Defaults (`wgm_v2_config.py`) follow Appendix C.1: gamma = 0.1, M = 3 search
tasks per new child, one BAI evaluation per iteration, Wilson intervals with
z = 1.96, and B_fin = floor(0.325 B) of the total budget B reserved for final
selection (260 of 800). This repository also includes the HGM and DGM
baselines.

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
