"""DGM legacy functions, from upstream a565fd2 (Apache-2.0).

Only change to selection: an injectable Random with the same Python semantics.
Do not repair the best sorting, eligible-parent exception or unresolved_ids == 0.
"""
import math
import os
from utils.common_utils import load_json_file
from prompts.self_improvement_prompt import find_selfimprove_eval_logs

def any_exceeding_context_length(output_dir, commit_id, instance_ids):
    """
    Check if any of the issues have exceeded the context length.
    """
    for instance_id in instance_ids:
        md_logs, _, _, _ = find_selfimprove_eval_logs(instance_id, output_dir, commit_id, filter=False)
        md_log = md_logs[0]
        error_str = "Error in get_response_withtools: Error code: 400 - {'message': 'Input is too long for requested model.'}"
        # Repeated error_str means no attempt to fix it
        if f'{error_str}\n{error_str}' in md_log:
            return True
    return False

def choose_selfimproves(output_dir, archive, selfimprove_size, method='random', run_baseline=None, polyglot=False, *, rng=None):
    """
    Choose self-improve attempts for the current generation.
    """
    random = rng if rng is not None else __import__("random")
    selfimprove_entries = []

    # Get parent candidates
    candidates = {}
    for commit in archive:
        try:
            metadata_path = os.path.join(output_dir, commit, "metadata.json")
            metadata = load_json_file(metadata_path)
            candidates[commit] = {
                'accuracy_score': metadata['overall_performance']['accuracy_score'],
                'total_unresolved_ids': metadata['overall_performance']['total_unresolved_ids'],
                'total_emptypatch_ids': metadata['overall_performance']['total_emptypatch_ids'],
                'total_resolved_ids': metadata['overall_performance']['total_resolved_ids'],
                'children_count': 0,
            }
            # update children count, parent should already be in the archive
            if commit != 'initial':
                parent_commit = metadata['parent_commit']
                candidates[parent_commit]['children_count'] += 1
        except Exception as e:
            # probably because swe-eval failed, generated code did not compile, etc.
            print(f"{commit} not eligible for being a parent: {e}")
            continue

    # Choose parents based on method and baseline
    if run_baseline == 'no_darwin':
        # Always take the last commit
        commits = list(candidates.keys())
        parent_commits = commits[-1:]
    elif method == 'score_prop':
        # Choose parents based on score
        commits = list(candidates.keys())
        scores = [candidates[commit]['accuracy_score'] for commit in commits]
        scores = [1 / (1 + math.exp(-10*(score-0.5))) for score in scores]
        probabilities = [score / sum(scores) for score in scores]
        print(commits)
        parent_commits = random.choices(commits, probabilities, k=selfimprove_size)
    elif method == 'score_child_prop':
        # Choose parents based on score and the number of children
        commits = list(candidates.keys())
        scores = [candidates[commit]['accuracy_score'] for commit in commits]
        scores = [1 / (1 + math.exp(-10*(score-0.5))) for score in scores]
        children_counts = [candidates[commit]['children_count'] for commit in commits]
        children_counts = [1 / (1 + count) for count in children_counts]
        probabilities = [score * count for score, count in zip(scores, children_counts)]
        probabilities = [prob / sum(probabilities) for prob in probabilities]
        parent_commits = random.choices(commits, probabilities, k=selfimprove_size)
    elif method == 'best':
        # Choose parents with the best score
        sorted_commits = sorted(candidates, key=lambda x: candidates[x]['accuracy_score'])
        parent_commits = sorted_commits[:min(selfimprove_size, len(sorted_commits))]
        if len(parent_commits) < selfimprove_size:
            parent_commits.extend(random.choices(parent_commits, k=selfimprove_size - len(parent_commits)))
    else:
        # Choose parents randomly
        parent_commits = random.choices(list(candidates.keys()), k=selfimprove_size)

    # Choose entries for each parent
    for parent_commit in parent_commits:
        empty_ids = candidates[parent_commit]['total_emptypatch_ids']
        resolved_ids = candidates[parent_commit]['total_resolved_ids']
        unresolved_ids = candidates[parent_commit]['total_unresolved_ids']

        if polyglot:
            entry_ids = empty_ids + unresolved_ids
            if not entry_ids:
                entry_ids = resolved_ids + empty_ids + unresolved_ids
        else:
            num_total_ids = len(empty_ids) + len(resolved_ids) + len(unresolved_ids)

            # Solve empty patches
            if len(empty_ids) >= 0.1 * num_total_ids and random.random() < 0.25:
                entry = 'solve_empty_patches'
                selfimprove_entries.append((parent_commit, entry))
                continue

            # Solve stochasticity
            if random.random() < 0.25:
                entry = 'solve_stochasticity'
                selfimprove_entries.append((parent_commit, entry))
                continue

            # Solve context length
            if any_exceeding_context_length(output_dir, parent_commit, empty_ids + unresolved_ids) and \
                random.random() < 0.25:
                entry = 'solve_contextlength'
                selfimprove_entries.append((parent_commit, entry))
                continue

            # Choose a random unresolved entry
            if unresolved_ids == 0:
                continue
            entry_ids = unresolved_ids
        entry = random.choice(entry_ids)
        selfimprove_entries.append((parent_commit, entry))

    return selfimprove_entries

def get_original_score(output_dir):
    """
    Get the original score from the initial version.
    """
    metadata = load_json_file(os.path.join(output_dir, "initial", "metadata.json"))
    return metadata["overall_performance"]["accuracy_score"]

def update_archive(output_dir, archive, new_ids, method='keep_all', noise_leeway=0.1):
    """
    Update the archive with the new self-improve runs.
    """
    if method == 'keep_better':
        # keep only better ones
        original_score = get_original_score(output_dir) - noise_leeway
        for run_id in new_ids:
            metadata = load_json_file(os.path.join(output_dir, run_id, "metadata.json"))
            score = metadata["overall_performance"]["accuracy_score"]
            if score >= original_score:
                archive.append(run_id)
    else:
        # keep everything
        archive += new_ids

    return archive

def is_compiled_self_improve(metadata, num_swe_issues=[], logger=None):
    """
    Checks if the run was properly compiled and 'self-improved' by verifying:
      1. The 'overall_performance' dict has the required keys:
         ('accuracy_score', 'total_unresolved_ids', 'total_resolved_ids', 'total_emptypatch_ids').
      2. There is at least one non-empty patch (resolved + unresolved > 0).
      3. If num_swe_issues is provided, the total number of evaluated issues matches num_swe_issues.

    Returns True if all conditions are met, else False.
    """
    overall_perf = metadata.get('overall_performance', {})
    required_keys = ['accuracy_score', 'total_unresolved_ids', 'total_resolved_ids', 'total_emptypatch_ids']

    # 1. Must have the required keys
    if not overall_perf or not all(k in overall_perf for k in required_keys):
        logger.info(f"no required keys")
        return False

    # 2. Must have at least one non-empty patch
    num_resolved = len(overall_perf['total_resolved_ids'])
    num_unresolved = len(overall_perf['total_unresolved_ids'])
    if (num_resolved + num_unresolved) == 0:
        logger.info(f"no non-empty patch")
        return False

    # 3. If specified, total evaluated must match num_swe_issues, else it means that some didn't compile
    total_evaluated = overall_perf['total_submitted_instances']
    if total_evaluated < num_swe_issues[0]:
        logger.info(f"not match num_issues")
        return False

    return True


def best_belief(archive, metadata):
    return max(archive, key=lambda candidate: metadata[candidate]["overall_performance"]["accuracy_score"])
