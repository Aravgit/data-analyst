# Git Hooks

This repo uses shared git hooks in `.githooks`.

Enable them locally:

```bash
git config core.hooksPath .githooks
```

## Pre-commit checks
- Block commits on `main`
- Enforce branch naming: `feature|fix|chore|docs|refactor|test|perf|hotfix/<slug>`
- Block large files over 50MB
- Block committing `node_modules`
- Fail on merge conflict markers
- Fail on whitespace errors

## Pre-push checks
- Block pushes on `main`

