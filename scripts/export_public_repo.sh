#!/usr/bin/env bash
# Make a clean, history-free copy of the deployable files for a NEW public repository.
#   scripts/export_public_repo.sh ../power-peak-public
# It only writes to the target folder: it never touches a remote. The copy has one commit, so none of the
# private repository's history (earlier commits, author details, deleted files) is published.
set -euo pipefail
target="${1:?usage: export_public_repo.sh <new-folder>}"
[ -e "$target" ] && { echo "$target already exists" >&2; exit 1; }
[ -z "$(git status --porcelain)" ] || { echo "commit or stash your changes first" >&2; exit 1; }
mkdir -p "$target"
git archive HEAD -- . ':(exclude)RESUME.md' | tar -x -C "$target"   # tracked files only (ignored data/logs/raw never leave); RESUME.md = private notes
cd "$target"
git init -q -b main
git add -A
# neutral author: the private repository's git identity (name, e-mail) is not published. Set
# EXPORT_NAME / EXPORT_EMAIL to use your own on purpose.
git -c user.name="${EXPORT_NAME:-Power Peak AI}" -c user.email="${EXPORT_EMAIL:-noreply@users.noreply.github.com}" \
    commit -q -m "Power Peak AI: public read-only snapshot"
echo "Created $(pwd): $(git ls-files | wc -l) files, $(du -sh .git | cut -f1) of history."
echo "Next: follow DEPLOY.md (create an empty PUBLIC repository on GitHub, then add it as 'origin' and push)."
