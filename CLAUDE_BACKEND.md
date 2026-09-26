# Claude Code CLI backend (optional)

The repository supports the **Claude Code CLI** (`claude -p`, account login) as a
second host agent backend. It is not a direct Claude API backend: no API key,
SDK client or self-implemented tool loop is involved; the agent loop and tools
are those of the Claude Code CLI. The Codex
backend is the default provider.

Claude results form a **separate model condition**. They must not be merged with
Codex runs, and no Codex root observation, cache, tree or final binding may be
imported into a Claude run: the Claude root is evaluated separately.

## Selecting the backend

```yaml
codex:
  provider: claude_code   # default is codex
  model: <model>          # or pass --codex-model; formal runs add --require-pinned-model
  reasoning_effort: null  # low | medium | high | xhigh | max
```

The section keeps the historical name `codex`, and the `llm.*_backend` value
`codex_exec` keeps meaning "host agent CLI pipeline". **Every** phase using
that pipeline (diagnosis, self-improvement candidate generation, downstream
solving, BAI/final evaluation) therefore uses the one configured provider;
a run cannot be silently mixed. The `legacy_api` backends are unaffected and
not used.

Implementation: `ClaudeCodeConfig` subclasses `CodexExecConfig`, and
`CodexExecRunner(config)` returns a `ClaudeCodeRunner` when
`config.provider == "claude_code"`. The runner replaces only argv, the child
environment and final-message extraction. Process start/terminal evidence,
timeouts, process-group termination, redaction, ledger callbacks and
exactly-once semantics are the shared Codex code paths.

## Tool protocol and differences from `codex exec`

| Aspect | Codex (`codex exec`) | Claude (`claude -p`) |
| --- | --- | --- |
| Prompt | stdin | stdin |
| Transcript | `--json` JSONL | `--output-format stream-json --verbose` |
| Final message | `--output-last-message` file | `result.result` event field |
| Structured output | `--output-schema FILE` | `--json-schema JSON`, read from `result.structured_output` |
| No session state | `--ephemeral` | `--no-session-persistence` |
| Model / effort | `--model`, `-c model_reasoning_effort` | `--model`, `--effort` (adds `max`, no `minimal`) |
| Customization | user Codex config | `--safe-mode --setting-sources "" --strict-mcp-config`: no CLAUDE.md, hooks, plugins, skills, MCP or settings |
| Write-mode tools | Codex built-ins | allowlist `Bash, Read, Edit, Write, Glob, Grep, TodoWrite`; no web, no subagents |
| Read-only tools | `--sandbox read-only` | allowlist `Read, Glob, Grep`, `--permission-mode dontAsk`, read-only mounts |
| Approvals | `--approve-for-me` | `--permission-mode bypassPermissions`, inside the confinement below |
| OS isolation | Codex sandbox (workspace-write) | bubblewrap around the whole CLI, see *Exposure boundary* |
| Command network | Codex workspace-write default | **not restricted**: Bash commands share the host network, which the CLI itself needs |
| Turn limit | none; bounded by timeout | none; bounded by the same timeout |
| Usage | `turn.completed.usage` | final `result.usage`; `input_tokens` = uncached + cache writes + cache reads, `cached_input_tokens` = cache reads |
| Observed model | lifecycle `model` fields | `system/init.model`; `billed_models` lists auxiliary models in `modelUsage` |
| Cost | not computed | `cost_usd` records the CLI list-price figure; subscription usage is not an invoice |

The Claude Code built-in Bash sandbox is deliberately not used. It needs
`socat`, and without it the CLI prints a warning and runs commands unsandboxed:
a command could then write outside the workspace. With `confinement: bwrap`
a missing `bwrap` fails before the process starts. `confinement: none` exists
only for offline tests and is recorded in the fingerprint.

## Exposure boundary (bwrap)

| Resource | Inside the confinement |
| --- | --- |
| Host filesystem | read-only bind of `/`; writes fail with `Read-only file system` |
| Workdir | writable in write mode, read-only for diagnosis; `--add-dir` directories likewise |
| `/tmp` | private tmpfs; `TMPDIR=/tmp` |
| Processes | private PID namespace, killed with bwrap on timeout |
| Docker / containerd / podman sockets | the socket directory (`/run`) is replaced by a tmpfs and every non-runtime entry mounted back; `docker ps` cannot connect |
| `$HOME` | tmpfs; only the top-level entries needed by `PATH` and the CLI binary (for example `~/.local` or a Conda installation) are mounted back read-only. `~/.ssh`, `~/.codex`, cloud and Git-host credentials are absent |
| Claude login | `~/.claude` and `~/.claude.json` writable (OAuth refresh); settings, CLAUDE.md, hooks, agents, commands, skills, plugins read-only; other sessions' transcripts and history (`projects`, `history.jsonl`, ...) replaced by empty mounts |
| `hidden_paths` | configured paths (for example `results/` and the benchmark datasets) are emptied in this checkout (earlier results, gold patches); a workdir inside `results/` is mounted back on top |
| Network | **shared with the host** (required by the CLI); commands can reach the internet |

Residual exposure: the Claude OAuth credentials are readable by tools (the CLI
needs them), and the network is open, so a misbehaving model could exfiltrate
that token. Other host files readable by the user outside the hidden paths
remain readable, as under Codex workspace-write.

The evolution boundary is unchanged: the only evolvable downstream policy is
`prompts/downstream_instructions.txt`. `--safe-mode` also prevents an evolved
candidate from adding a CLAUDE.md policy channel.

## Identity

`asdict(config.codex)` includes `provider`, the tool allowlists and the
confinement, so the effective configuration, `configuration_sha256` and the
reproducibility fingerprint of a Claude run differ from any Codex run.
`codex_cli_version` records `claude --version`; the manifest `codex` block also has `provider`.
Each call record under `codex_calls/` has `provider: claude_code`.

## Account quota (429) handling

Claude Code CLI runs on the account's rolling usage windows. Two rules keep an
exhausted window from being recorded as a scientific failure:

1. **Pre-launch wait.** Each call's `rate_limit_event` reports the window's
   utilization and reset time. When the last observed utilization is at least
   `quota_wait_threshold` or the status is `rejected`, the next
   launch waits until the reset. Nothing has started while waiting.
2. **Rejected without side effects = not started.** If an attempt ends in a
   429 / limit rejection, was not timed out, and used only side-effect-free
   tools (`Read`, `Glob`, `Grep`, `TodoWrite`, `StructuredOutput`), the
   workspace and host are unchanged. The attempt is discarded, the runner waits
   for the reset and relaunches the same call (at most
   `rate_limit_relaunches` times). The ledger sees one call: one start
   (the first process) and the terminal of the final process. Each discarded
   attempt is recorded in the call record's `rate_limit_rejections` (status,
   tools, turns, output tokens, quota window, reset); waiting time is excluded
   from `duration_seconds`.

A rejection after any other tool (Bash, Edit, Write) is a real, failed call and
is classified as before (execution error); it is never retried. If waiting is
interrupted, the rejected attempt's real terminal is delivered first. The
account is shared with any interactive Claude session of the same user, which
also consumes the window.

## Claude condition root

`root_cache.compatibility` adds `provider` and `prompt_identity` for Claude
roots. HGM (`comparison_root`) and DGM (`dgm_legacy_search.root_condition_matches`)
read the condition (provider, model, reasoning effort, prompt identity) from the
frozen run configuration / manifest and require the root to match.
The Claude root is evaluated on all benchmark tasks with Claude; no Codex
observation is imported.

## Prompts

All backend-dependent prompt sentences are in `provider_prompts.py`. For Codex
(`provider` absent) they are byte-identical to the original prompts, and Codex
manifests keep their historical fields. Under `claude_code` the diagnosis
runtime-boundary sentence, the bounded diagnosis system text and the
candidate-generation policy note use provider-neutral wording ("one host agent
CLI call", "The downstream agent backend"), revision `provider-neutral-v1`.
The manifest `codex.prompt_identity` records the digest of these texts and the
fingerprint includes it as `provider_prompt_identity`. HGM, DGM and WGM all
build prompts through the same functions (`search_operations.diagnosis_prompt`,
`candidate_workspace`, `search_operation_workspace`), so under Claude the three
algorithms share one frozen prompt.

The runner strips nested-session markers, so it also works when launched from
inside a Claude Code session; formal runs should still use a plain shell.

## Pre-launch retries and CLI version

A baseline task that failed before any model launch (zero calls, stage
`dataset_image_preparation` or `task_workspace_extraction`) can be re-run in
place: the failure is moved to
`<run>/evidence/prelaunch-retries/<task>/attempt-N/` with a digest record and the
original command is resumed on the same commit. `root_cache.validate_baseline` accepts such a task
out of execution order only with valid evidence (digests, pre-launch stage,
zero calls, original submission time), and accepts a baseline commit that is an
ancestor of the importing commit; all scientific compatibility fields are still
compared exactly. The evidence is copied into the root cache.

Claude CLI self-updates in interactive use. Pin one CLI version (for example a
copied binary first on `PATH`) for all runs of a condition, since
`codex_cli_version` is part of the fingerprint and root identity.

## Operator-authorized recoveries

All are explicit operator commands; each keeps the failed evidence (events,
call records) and writes an audit artifact. None records a failure as an
unresolved task, and all require the run's own commit.

- `search-authorize-diagnosis-retry`: a failed read-only diagnosis call
  (Read/Glob/Grep only) is abandoned (`diagnosis_abandoned` event) and counts
  as one used diagnosis attempt, as in the original HGM rule. Typical
  causes are a login expiry during a quota wait and
  `error_max_structured_output_retries` (Claude Code validates the diagnosis
  JSON schema client-side).
- `search-authorize-evaluation-retry`: a search evaluation whose downstream
  call failed on a connection error (`terminal_reason=api_error`, no HTTP
  status) after using workspace tools is abandoned (`attempt_abandoned`,
  reason `audited_connection_failure_retry`) and runs once more as a fresh
  attempt. Readers use the latest binding.
