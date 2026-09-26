"""DGM SWE diagnosis text/250k truncation, upstream a565fd2 (Apache-2.0).

The imported constants/code reader are unchanged from the same upstream source.
Host Codex policy/schema wrapping is performed by the common transport adapter.
"""
from prompts.self_improvement_prompt import (
    coding_agent_summary, swe_issue_prompt, diagnose_prompt, diagnose_prompt_emptypatches,
    diagnose_prompt_stochasticity, diagnose_prompt_contextlength,
    find_selfimprove_eval_logs, get_current_code)

diagnose_system_message = """Here is the implementation of the coding agent.

# Coding Agent Implementation
----- Coding Agent Implementation Start -----
{code}
----- Coding Agent Implementation End -----

Your task is to identify ONE detailed plan that would improve the agent's coding ability. The improvement should not be specific to any particular GitHub issue or repository.
"""

def process_selfimprove_eval_logs(md_logs, eval_logs, predicted_patches, eval_results):
    # NOTE: using only the first logs
    md_log = md_logs[0] if md_logs else "No logs available."
    eval_log = eval_logs[0] if eval_logs else "No test results available. Assume all tests failed."
    predicted_patch = predicted_patches[0] if predicted_patches else "No predicted patch available. Assume the agent failed."

    # truncate logs if too long
    if len(md_log) > 250000:
        md_log = md_log[:250000] + "\n<log clipped>"

    eval_result = eval_results[0] if eval_results else "No evaluation result available. Assume the agent failed."
    return md_log, eval_log, predicted_patch, eval_result

def get_diagnose_prompt_swe(entry_id, commit, root_dir, out_dir, dataset, patch_files=[]):
    if entry_id == 'solve_empty_patches':
        # Get user prompt for solving empty patches
        diagnose_prompt_out = diagnose_prompt_emptypatches
    elif entry_id == 'solve_stochasticity':
        # Get user prompt for solving stochasticity
        diagnose_prompt_out = diagnose_prompt_stochasticity
    elif entry_id == 'solve_contextlength':
        # Get user prompt for solving context length
        diagnose_prompt_out = diagnose_prompt_contextlength
    else:
        # Get user prompt for the entry
        md_logs, eval_logs, predicted_patches, eval_results = find_selfimprove_eval_logs(entry_id, out_dir, commit_id=commit)
        md_log, eval_log, predicted_patch, eval_result = process_selfimprove_eval_logs(md_logs, eval_logs, predicted_patches, eval_results)
        entry = next((e for e in dataset if e['instance_id'] == entry_id), None)
        answer_patch = entry['patch']
        test_patch = entry['test_patch']
        github_issue = entry['problem_statement']
        diagnose_prompt_out = swe_issue_prompt + diagnose_prompt.format(md_log=md_log, eval_log=eval_log, predicted_patch=predicted_patch, answer_patch=answer_patch, test_patch=test_patch, github_issue=github_issue)

    # Get system prompt
    code_files = ['coding_agent.py', 'tools/', 'utils/']
    exclude_files = [
        'utils/evo_utils.py',
        'utils/docker_utils.py',
        'utils/swe_log_parsers.py',
        'prompts/self_improvement_prompt.py',
    ]
    code_text = get_current_code(root_dir, code_files, patch_files=patch_files, exclude_files=exclude_files)
    diagnose_system_message_out = coding_agent_summary + diagnose_system_message.format(code=code_text)

    return diagnose_system_message_out, diagnose_prompt_out
