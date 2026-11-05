# Contributing to SRL

Thanks for your interest in improving SRL! Bug reports, docs fixes, new
environment configs and algorithm improvements are all welcome.

By participating you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md).
Security issues: please follow [SECURITY.md](SECURITY.md) instead of opening a
public issue.

## Development setup

SRL supports Python 3.10 – 3.13.

```bash
git clone https://github.com/Bigkatoan/SRL.git
cd SRL
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Optional but recommended on machines without a GPU: the CPU-only torch wheel
# is ~10x smaller than the default CUDA build. CI does exactly this.
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -e ".[dev]"            # test + lint + docs + tracking tools
pip install -e ".[mujoco,robotics]"  # optional: sim backends used by some tests
pre-commit install                 # run ruff/black/yaml checks on every commit
```

The `dev` extra is the union of the `test`, `lint`, `docs` and `tracking`
extras; install a subset (e.g. `pip install -e ".[test]"`) if you prefer.

## Running tests

The default suite is fast and CPU-only. It is what the `Tests` workflow runs on
Linux (Python 3.10 – 3.13), macOS and Windows:

```bash
pytest -m "not slow and not gpu"
# with coverage (CI enforces fail_under = 65 % from pyproject.toml)
pytest -m "not slow and not gpu" --cov=srl --cov-report=term
```

Test markers (declared in `pyproject.toml`, enforced with `--strict-markers`):

| Marker        | Meaning                                                              | Where it runs |
|---------------|----------------------------------------------------------------------|---------------|
| `slow`        | End-to-end learning regressions (`tests/test_learning.py`), minutes each | nightly `learning` job, manual dispatch, or PRs labelled `run-slow` |
| `gpu`         | Needs CUDA; skipped automatically without a GPU                       | locally only  |
| `integration` | Needs an external simulator/runtime (Isaac Lab, mjlab, ROS 2, …)     | locally only  |

```bash
pytest -m slow --timeout=3600       # learning regressions (CPU, ~tens of minutes)
pytest tests/test_configs_parse.py  # every configs/**/*.yaml builds + matches its env
```

Each test has a 300 s timeout (`pytest-timeout`). If you add a test that
legitimately needs longer, mark it `slow`.

### Config sweep

`scripts/run_full_matrix_benchmark.sh` trains every config under
`configs/envs/` and judges the results against
`configs/benchmarks/*_targets.yaml`. It is a manual tool (not run in CI):

```bash
bash scripts/run_full_matrix_benchmark.sh --python "$(command -v python)" --label quick_check
bash scripts/run_full_matrix_benchmark.sh --help   # all options
```

## Code style

- **Lint:** `ruff check srl tests examples templates` (rule set pinned in
  `pyproject.toml`: E/F/I/UP/B).
- **Formatting:** `black --check srl tests examples templates`
  (line length 100). Tool versions are pinned in the `lint` extra and in
  `.pre-commit-config.yaml`; bump them deliberately in one PR.
- **Type checking:** `mypy srl` runs in CI in *non-blocking* mode while the
  existing baseline (~281 errors) is worked down. Please don't add new errors
  in code you touch.
- Type hints on public functions; NumPy-style or Google-style docstrings.

### Formatting: planned migration to `ruff format`

`ruff format` is configured in `pyproject.toml` to match black, but black is
still the enforced formatter: switching would reformat ~16 files in `srl/` and
`tests/`, which is deferred to avoid conflicts with in-flight PRs. You can
preview it with `pre-commit run ruff-format --hook-stage manual --all-files`.
When the migration lands, `lint.yml` and `.pre-commit-config.yaml` swap black
for `ruff format --check`.

## Building the docs

```bash
python -m sphinx -b html -W --keep-going -d docs/_build/doctrees docs docs/_build/html
```

Warnings are errors in CI (`-W`), so fix any broken cross-references before
pushing. Docs are deployed to <https://bigkatoan.github.io/SRL> from `main`.

## Adding a new environment

1. Check whether it needs a custom wrapper (goal-based → `GoalEnvWrapper`,
   Isaac Lab / mjlab → `IsaacLabWrapper`).
2. Add `configs/envs/<env>_<algo>.yaml` with `env_id`, `env_type` and `algo`.
   `tests/test_configs_parse.py` picks it up automatically: it must build, run
   a dummy forward pass and (if the env's runtime is installed) match the
   env's observation/action dims.
3. Optionally add a target to `configs/benchmarks/*_targets.yaml`.
4. Add a docs page under `docs/source/environments/` and link it from its
   `index.md`.

## Adding a new algorithm

1. Create `srl/algorithms/<name>.py` extending `BaseAgent`.
2. Add a `<Name>Config` dataclass in `srl/core/config.py`.
3. Register losses in `srl/losses/rl_losses.py` if needed.
4. Wire it into `srl/cli/train.py` and add unit tests.
5. Add a learning regression to `tests/test_learning.py` (`@pytest.mark.slow`).
6. Document it in `docs/source/algorithms.md`.

## Pull requests

- Branch from `main`; keep PRs focused (one logical change).
- Add an entry under `## [Unreleased]` in [CHANGELOG.md](CHANGELOG.md).
- Make sure lint, tests and the docs build pass locally.
- For changes that can affect learning performance, add the `run-slow` label
  to run the learning regression job on your PR.

Releases are cut by maintainers; see [RELEASING.md](RELEASING.md).
