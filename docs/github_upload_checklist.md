# GitHub Upload Checklist

Use this checklist before pushing changes to GitHub.

## 1) Local quality checks

```bash
python -m compileall -q .
pytest
```

## 2) Verify tracked files

- No local cache files in staged changes (`__pycache__`, `.DS_Store`, `.pyc`).
- No temporary outputs accidentally staged.
- PR template and docs are updated if workflow changed.

## 3) Commit in clear units

Examples:

- `chore(repo): add CI and dependency manifests`
- `test(data): add processor unit tests`
- `docs(readme): standardize setup and dev checks`

## 4) Push and open PR

```bash
git push -u origin <your-branch>
```

PR should include:

- Why this change is needed
- What changed
- Validation commands used
- Any data path or behavior impacts
