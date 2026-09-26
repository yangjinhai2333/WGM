import argparse
import datetime
from copy import copy, deepcopy
from dataclasses import replace
import json
import logging
import os
import random
import re
import subprocess

from config import validate_diagnose_backend
from prompts.self_improvement_prompt import (get_diagnose_prompt_polyglot,
                                             get_diagnose_prompt_swe,
                                             get_problem_description_prompt)
from runners import (CodexConfigurationError, CodexExecConfig, CodexExecRequest,
                     CodexExecRunner, CodexExecutionError, CodexOutputError,
                     CodexRunnerError, CodexStartError)

# Existing compatibility state: sample_child supplies dataset; direct legacy
# callers may still set diagnose_llm. Backend/runner settings are explicit args.
dataset = None
diagnose_llm = ""
self_improve_llm = ""
timeout = 3600
n_evals = 0


DIAGNOSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "implementation_suggestion": {"type": "string", "minLength": 1},
        "problem_description": {"type": "string", "minLength": 1},
    },
    "required": ["implementation_suggestion", "problem_description"],
    "additionalProperties": False,
}


class DiagnosisValidationError(ValueError):
    """A diagnostic response cannot be converted into a problem statement."""


def validate_diagnosis(value, *, allow_extra_fields=False):
    """Validate locally without rewriting the two strings used by the formatter."""
    if not isinstance(value, dict):
        raise DiagnosisValidationError("response must be a JSON object")
    required = DIAGNOSIS_SCHEMA["required"]
    if any(field not in value for field in required):
        raise DiagnosisValidationError("missing required fields")
    if not allow_extra_fields and set(value) != set(required):
        raise DiagnosisValidationError("unexpected response fields")
    for field in required:
        if not isinstance(value[field], str):
            raise DiagnosisValidationError("required fields must be strings")
        if not value[field].strip():
            raise DiagnosisValidationError("required fields must not be empty or whitespace")
    return value


def _diagnosis_runner(runner, codex_config):
    if runner is None:
        return CodexExecRunner(replace(
            codex_config if codex_config is not None else CodexExecConfig(),
            sandbox="read-only", approve_for_me=False, output_schema=None,
        ))
    # Preserve injected runners and their instrumentation, without mutating a
    # shared CLI runner's config. Pure test doubles only need run_json().
    if isinstance(getattr(runner, "config", None), CodexExecConfig):
        runner = copy(runner)
        runner.config = replace(runner.config, sandbox="read-only", approve_for_me=False,
                                output_schema=None)
    return runner


def diagnose_problem(
    entry, commit, root_dir, out_dir, patch_files=None, max_attempts=2, polyglot=False,
    *, backend="codex_exec", codex_config=None, runner=None, legacy_model=None, log=None,
    downstream_backend="legacy_api", recorder=None,
    context_config=None,
):
    """Return the original problem-statement format, or None after failures.

    Compatibility: max_attempts is the *retry budget*, as in the old recursive
    implementation. Total attempts = max_attempts + 1 (default 3; zero means one).
    Each attempt rebuilds the original prompt, preserving Polyglot's existing
    randomized prompt choice. Only the legacy_api branch imports the API client.
    The caller owns scheduling; this function does not change HGM worker counts.
    """
    validate_diagnose_backend(backend)
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 0:
        raise ValueError("max_attempts must be a nonnegative integer retry budget.")
    root_dir = os.path.abspath(root_dir)
    patch_files = [] if patch_files is None else patch_files
    log = log if log is not None else logging.getLogger(__name__).warning
    if backend == "codex_exec":
        runner = _diagnosis_runner(runner, codex_config)
    from provider_prompts import context_system, provider_of, runtime_boundary
    prompt_provider = provider_of(getattr(runner, "config", None))
    total_attempts = max_attempts + 1
    context_parts = None
    context_budget = None
    if context_config is not None:
        context_config.validate()
        context_budget = context_config.max_input_chars
    forbidden_prompts = set()
    for attempt in range(1, total_attempts + 1):
        phase = "prompt construction failed"
        try:
            if context_config is None:
                prompt_builder = get_diagnose_prompt_polyglot if polyglot else get_diagnose_prompt_swe
                system_message, user_prompt = prompt_builder(
                    entry, commit, root_dir, out_dir, dataset, patch_files=patch_files,
                )
            if context_config is None and downstream_backend == "codex_exec":
                from downstream_solver import load_agent_policy
                policy = load_agent_policy(root_dir, patch_files, polyglot=polyglot)
                system_message += (
                    runtime_boundary(prompt_provider) +
                    f"<current_downstream_instructions>\n{policy}\n</current_downstream_instructions>"
                )
            if backend == "codex_exec":
                phase = "CLI execution failed"
                context_stats = None
                if context_config is not None:
                    from diagnosis_context import build_parts, bounded_prompt
                    if context_parts is None:
                        context_parts = build_parts(entry, commit, root_dir, out_dir, dataset, patch_files, polyglot=polyglot)
                    prompt, context_stats = bounded_prompt(context_parts, context_budget,
                                                                    system=context_system(prompt_provider))
                    if context_stats["prompt_sha256"] in forbidden_prompts:
                        log("Diagnosis stopped: refusing to repeat a context-exhausted request.")
                        return None
                else:
                    prompt = (
                        f"<system_instructions>\n{system_message}\n</system_instructions>\n\n"
                        f"<diagnosis_request>\n{user_prompt}\n</diagnosis_request>\n\n"
                        "<response_format>\nReturn only a JSON object matching the supplied schema: "
                        "exactly implementation_suggestion and problem_description, both nonempty strings. "
                        "Preserve the proposed improvement in these fields. Do not emit Markdown fences "
                        "or the other intermediate analysis fields requested above.\n</response_format>"
                    )
                from runners.usage import invoke
                from network_readiness import await_runner
                receipt = await_runner(runner, recorder=recorder)
                response_json = invoke(runner,
                    CodexExecRequest(prompt=prompt, workdir=root_dir, readiness_receipt=receipt),
                    "diagnosis", recorder=recorder, schema=deepcopy(DIAGNOSIS_SCHEMA),
                    **({"context_stats": context_stats} if context_stats is not None else {}),
                ).value
            else:
                phase = "legacy API request failed"
                from llm import create_client, extract_json_between_markers, get_response_from_llm

                client, model = create_client(
                    diagnose_llm if legacy_model is None else legacy_model
                )
                response, _ = get_response_from_llm(
                    msg=user_prompt, client=client, model=model,
                    system_message=system_message, print_debug=False, msg_history=None,
                )
                response_json = extract_json_between_markers(response)
                if response_json is None:
                    raise CodexOutputError("Invalid legacy JSON response.")
            validate_diagnosis(response_json, allow_extra_fields=(backend == "legacy_api"))
            return get_problem_description_prompt(response_json, polyglot)
        except CodexExecutionError as error:
            if context_config is not None:
                from diagnosis_context import is_context_error, smaller_budget
                if is_context_error(error):
                    forbidden_prompts.add(context_stats["prompt_sha256"])
                    context_budget = smaller_budget(context_config, context_budget, prompt)
                    if context_budget is None:
                        log("Diagnosis stopped: context error and minimum safe input budget reached.")
                        return None
            reason = "CLI timed out" if error.result.timed_out else (
                f"CLI exited with status {error.result.returncode}"
            )
            if error.stderr_summary:
                reason += f"; stderr: {error.stderr_summary}"
            if error.event_error_summary:
                reason += f"; events: {error.event_error_summary}"
        except TimeoutError:
            reason = "diagnosis request timed out"
        except CodexStartError:
            reason = "CLI could not start"
        except (CodexOutputError, json.JSONDecodeError):
            reason = "invalid JSON response or missing final message"
        except DiagnosisValidationError as error:
            reason = str(error)  # Only fixed diagnostics created by our validator.
        except CodexConfigurationError:
            reason = "invalid runner configuration"
        except CodexRunnerError:
            reason = "CLI execution failed"
        except Exception:
            reason = phase
        log(f"Diagnosis attempt {attempt}/{total_attempts} failed: {reason}.")
    return None


def save_metadata(metadata, output_dir):
    from pathlib import Path
    from candidate_workspace import _atomic_write
    _atomic_write(Path(output_dir) / "metadata.json", json.dumps(metadata, indent=4).encode())
