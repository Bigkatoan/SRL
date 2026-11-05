## Summary

<!-- What does this PR change, and why? Link related issues: "Fixes #123". -->

## Type of change

- [ ] Bug fix
- [ ] New feature (algorithm / environment / config option)
- [ ] Performance / learning-quality change
- [ ] Docs only
- [ ] CI / packaging / tooling

## How was this tested?

<!-- Commands you ran, new tests added, learning curves for algorithm changes. -->

## Checklist

- [ ] `pytest -m "not slow and not gpu"` passes locally
- [ ] `ruff check srl tests examples templates` and `black --check srl tests examples templates` pass
- [ ] New/changed behaviour is covered by tests
- [ ] Docs updated (`docs/`) if user-facing behaviour or the public API changed
- [ ] Entry added under `## [Unreleased]` in `CHANGELOG.md`
- [ ] For changes that can affect learning: added the `run-slow` label (runs `tests/test_learning.py`)
