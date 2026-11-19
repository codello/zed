#!/usr/bin/env bash
# Updates the fork after an update of the base branch:
# 1. Rebases the tools branch and each feature branch onto the base branch.
#    A pull request branch keeps its base if its commits apply to the base branch without conflicts.
# 2. Creates the combined branch again: the base branch plus the commits of all feature branches.
#
# If a conflict stops the script, resolve the conflict, continue the rebase or
# cherry-pick, and then run the script again.
#
# Options:
#   --push  After a successful update, force-pushes the tools branch and the feature branches
#           that are different from their copies on the fork remote. The script does not push
#           branches that do not exist on the fork remote.

# Bash must read the full script before it starts, because the script file changes when the script switches branches.
main() {
    set -euo pipefail

    local base=main
    local combined=build
    local tools_branch=build-tools
    local fork_remote=fork

    # The script picks the commits into the combined branch in this order.
    # An entry that contains '*', '?' or '[' is a pattern. The branches that match a pattern are in alphabetical order.
    # A '*' does not match '/'.
    local branch_patterns=(
        'chore/*'
        'fix/*'
        'feat/*'
    )

    # Feature branches with open upstream pull requests. The entries use the same syntax as the branch list.
    local pull_request_patterns=(
        'feat/mcp-resources'
        'fix/audio-output'
    )

    local push=false argument
    for argument in "$@"; do
        case $argument in
            --push) push=true ;;
            *) die "unknown option '$argument'." ;;
        esac
    done

    local git_dir
    git_dir=$(git rev-parse --git-dir)

    if rebase_in_progress "$git_dir"; then
        die "a rebase is in progress. Finish it, then run the script again."
    fi
    if [[ -f "$git_dir/MERGE_HEAD" || -f "$git_dir/CHERRY_PICK_HEAD" ]]; then
        die "a merge or cherry-pick is in progress. Finish it, then run the script again."
    fi
    if ! git diff --quiet || ! git diff --cached --quiet; then
        die "the working tree has uncommitted changes. Commit them first."
    fi

    local matches branch
    local features=()
    matches=$(list_branches "${branch_patterns[@]}")
    while IFS= read -r branch; do
        [[ -z "$branch" || "$branch" == "$tools_branch" || "$branch" == "$combined" ]] && continue
        contains "$branch" "${features[@]}" || features+=("$branch")
    done <<<"$matches"
    [[ ${#features[@]} -gt 0 ]] || die "no branch matches the branch list."

    local pull_requests=()
    matches=$(list_branches "${pull_request_patterns[@]}")
    while IFS= read -r branch; do
        [[ -n "$branch" ]] && pull_requests+=("$branch")
    done <<<"$matches"

    local original
    original=$(git symbolic-ref --quiet --short HEAD || git rev-parse HEAD)

    for branch in "$tools_branch" "${features[@]}"; do
        if git merge-base --is-ancestor "$base" "$branch"; then
            echo "update-fork: '$branch' is up to date."
            continue
        fi
        if contains "$branch" "${pull_requests[@]}" && applies_cleanly "$base" "$branch"; then
            echo "update-fork: '$branch' applies without conflicts. It keeps its base."
            continue
        fi
        echo "update-fork: rebasing '$branch'."
        git rebase "$base" "$branch" || continue_rebase "$git_dir" "$branch"
    done

    echo "update-fork: creating '$combined' again."
    git checkout --quiet -B "$combined" "$base"
    for branch in "${features[@]}"; do
        echo "update-fork: picking '$branch'."
        git cherry-pick --empty=drop "$base..$branch" || continue_cherry_pick "$git_dir" "$branch"
    done

    git checkout --quiet "$original"

    if [[ "$push" == true ]]; then
        local changed=()
        for branch in "$tools_branch" "${features[@]}"; do
            git show-ref --verify --quiet "refs/remotes/$fork_remote/$branch" || continue
            [[ "$(git rev-parse "$branch")" == "$(git rev-parse "refs/remotes/$fork_remote/$branch")" ]] && continue
            changed+=("$branch")
        done
        if [[ ${#changed[@]} -gt 0 ]]; then
            echo "update-fork: pushing ${changed[*]} to '$fork_remote'."
            git push --force-with-lease --force-if-includes "$fork_remote" "${changed[@]}"
        else
            echo "update-fork: '$fork_remote' is up to date."
        fi
    fi

    echo "update-fork: done."
}

# rerere replays recorded conflict resolutions. autoUpdate stages them, so the script can continue without a stop.
git() {
    command git -c rerere.enabled=true -c rerere.autoUpdate=true "$@"
}

die() {
    echo "update-fork: $*" >&2
    exit 1
}

# Writes the local branches that match the patterns, one branch on each line.
list_branches() {
    local pattern
    for pattern in "$@"; do
        if [[ "$pattern" == *[*?[]* ]]; then
            git for-each-ref --format='%(refname:short)' "refs/heads/$pattern"
        else
            git show-ref --verify --quiet "refs/heads/$pattern" || die "branch '$pattern' does not exist."
            echo "$pattern"
        fi
    done
}

contains() {
    local value=$1 item
    shift
    for item in "$@"; do
        [[ "$item" == "$value" ]] && return 0
    done
    return 1
}

# Tests in memory if the commits of the branch apply to the base. Recorded rerere resolutions do not apply to this test.
applies_cleanly() {
    local status=0
    git replay --ref-action=print --onto "$1" "$1..$2" >/dev/null || status=$?
    case $status in
        0) return 0 ;;
        1) return 1 ;;
        *) die "cannot test if '$2' applies to '$1'." ;;
    esac
}

rebase_in_progress() {
    [[ -d "$1/rebase-merge" || -d "$1/rebase-apply" ]]
}

has_unmerged_files() {
    [[ -n "$(git diff --name-only --diff-filter=U)" ]]
}

# Continues the current rebase while rerere resolves all conflicts.
continue_rebase() {
    while rebase_in_progress "$1"; do
        if has_unmerged_files; then
            die "rebase conflict in '$2'. Resolve it, run 'git rebase --continue', then run the script again."
        fi
        if ! GIT_EDITOR=true git rebase --continue && ! has_unmerged_files; then
            die "the rebase of '$2' stopped. Examine 'git status', finish the rebase, then run the script again."
        fi
    done
}

# Continues the current cherry-pick while rerere resolves all conflicts.
continue_cherry_pick() {
    while [[ -f "$1/CHERRY_PICK_HEAD" || -d "$1/sequencer" ]]; do
        if has_unmerged_files; then
            die "conflict while picking '$2'. Resolve it, run 'git cherry-pick --continue', then run the script again. rerere records your resolution."
        fi
        if ! GIT_EDITOR=true git cherry-pick --continue && ! has_unmerged_files; then
            die "picking '$2' stopped. Examine 'git status', finish the cherry-pick, then run the script again."
        fi
    done
}

main "$@"; exit
