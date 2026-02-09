# Contributing Guide

This project uses a lightweight Git workflow focused on stable releases, traceable data changes, and reproducible analysis.

## Branch Strategy

- Keep `main` releasable at all times.
- Do all work in short-lived branches.
- Branch naming format:
  - `feature/<scope>-<short-desc>`
  - `bugfix/<scope>-<short-desc>`
  - `hotfix/<scope>-<short-desc>`
  - `chore/<scope>-<short-desc>`
  - `refactor/<scope>-<short-desc>`
  - `docs/<scope>-<short-desc>`
  - `test/<scope>-<short-desc>`

Examples:

- `feature/parser-add-batch-loader`
- `bugfix/plot-fix-intensity-scale`
- `chore/repo-clean-structure`

## Commit Convention

Use Conventional Commits:

- Format: `<type>(<scope>): <subject>`
- Common `type` values: `feat`, `fix`, `chore`, `refactor`, `docs`, `test`

Examples:

- `feat(parser): add batch import for 11.5eV data`
- `fix(plot): correct y-axis scaling for low intensity peaks`

## Pull Request Workflow

1. Sync local main:
   - `git switch main`
   - `git pull`
2. Create a branch from `main`.
3. Keep commits focused (one logical change per commit).
4. Push branch and open a PR.
5. Merge only after checks pass and review feedback is addressed.

Recommended merge method:

- Use squash merge to keep history clean.

## Data Management Rules

- Treat raw experiment files as immutable.
- Put generated outputs in dedicated output directories, not mixed with source code.
- Avoid committing local caches, temporary plots, and one-off debug files.
- For large raw files, prefer Git LFS.

## Repository Cleanup Rules

- Use non-breaking cleanup first (structure and docs, no behavior change).
- If moving files, use `git mv` to preserve history.
- For path changes, include a migration note in the PR.
- Do not combine heavy refactors with feature development in one PR.
