# Repo Keep List

This file defines what should stay in the repository and what can be cleaned
before commit.

## Keep (runtime core)

- `main.py` (thin entrypoint)
- `src/__init__.py`
- `src/bl03u_massspec/app_main.py`
- `src/bl03u_massspec/massspec_func.py`
- `src/bl03u_massspec/massspec.py`
- `src/bl03u_massspec/data_processor.py`
- `src/bl03u_massspec/FindPeak.py`
- `icons/icon.png`
- `icons/bjt.png`

## Keep (engineering and collaboration)

- `.github/workflows/ci.yml`
- `.gitignore`
- `.editorconfig`
- `requirements.txt`
- `requirements-dev.txt`
- `pytest.ini`
- `tests/conftest.py`
- `tests/test_data_processor.py`
- `README.md`
- `CONTRIBUTING.md`
- `docs/github_upload_checklist.md`
- `docs/repository_reorganization.md`

## Keep (minimal sample data tracked by git)

- `data/raw/C6F11O2H/11.5eV/C23072502-0000.txt`

## Optional (not in main runtime path)

Move to `scripts/legacy/` if still useful, or remove if no longer needed:

- `src/bl03u_massspec/MsCal.py`
- `src/bl03u_massspec/caculate.py`
- `src/bl03u_massspec/peak_gaussain.py`
- `src/bl03u_massspec/data_processor_example.py`
- `src/bl03u_massspec/login_dialog.py`

## Safe to clean before commit

- `.DS_Store`
- `.idea/`
- `__pycache__/`
- `.pytest_cache/`
- `image/Temperature/`
- `path_to_save_images/`
- `data/abc.txt`
- `data/data_output.xlsx`
- `data/*.png`
- Old legacy files (if present): old root-level UI/generated files and old
  duplicate root-level modules.

## Automation

Use `scripts/safe_cleanup.sh` for dry-run/apply cleanup with backup.
