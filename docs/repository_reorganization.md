# Repository Reorganization Plan

This document defines a non-breaking, phased cleanup plan for this repository.

## Goals

- Keep `main` always runnable.
- Separate code, raw data, and generated outputs.
- Make experiment processing reproducible and reviewable.

## Current Challenges

- Source code and experiment files are mixed in top-level and `src/`.
- Generated artifacts and caches are stored in versioned paths.
- Review scope is hard to track when structure and feature work are mixed.

## Target Layout

```text
repo/
  src/                  # Application and analysis code
  scripts/              # CLI helpers and migration scripts
  data/
    raw/                # Immutable raw experiment files
    interim/            # Temporary transformed data
    processed/          # Stable processed datasets
  results/              # Plots and export outputs
  tests/                # Unit and regression tests
  docs/                 # Project and workflow docs
```

## Phase 1 (Now): Baseline Governance

- Add contribution rules and PR template.
- Add `.gitignore` for caches and generated outputs.
- Do not move runtime paths yet.

Exit criteria:

- Team follows branch and commit naming rules.
- New generated files no longer pollute PRs.

## Phase 2: Controlled Structure Migration

Run in dedicated branch: `chore/repo-clean-structure`.

Recommended move order:

1. Create target directories (`scripts`, `tests`, `results`, `data/raw`, `data/interim`, `data/processed`).
2. Move experiment datasets currently under `src/` into `data/raw/` using `git mv`.
3. Keep temporary compatibility path handling in code where needed.
4. Move ad hoc scripts into `scripts/` and update references.

Notes:

- Prefer multiple small PRs instead of one large migration PR.
- Avoid mixing feature logic changes with path migration.

## Phase 3: Path Contract in Code

- Define a single data-root configuration entry.
- Replace hardcoded relative paths with the shared data-root.
- Add startup validation for missing data directories.

Suggested checks:

- App startup still works.
- One raw dataset can be loaded from new location.
- Existing plotting and fitting still produce expected outputs.

## Phase 4: Reproducibility and CI

- Add minimal CI checks for lint and smoke run.
- Add regression samples in `tests/fixtures` (small files only).
- Require validation steps in every PR.

## Migration Safety Rules

- Use `git mv` for all file moves to preserve history.
- Keep each migration PR focused on one area.
- Document path changes in PR description.
- If a migration breaks runtime behavior, revert only that migration PR.
