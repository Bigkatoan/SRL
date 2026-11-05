# Contributing

The full contributor guide lives in
[CONTRIBUTING.md](https://github.com/Bigkatoan/SRL/blob/main/CONTRIBUTING.md)
in the repository root; release steps are in
[RELEASING.md](https://github.com/Bigkatoan/SRL/blob/main/RELEASING.md).

## Quick reference

```bash
git clone https://github.com/Bigkatoan/SRL.git && cd SRL
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # optional, CPU-only
pip install -e ".[dev]"
pre-commit install

pytest -m "not slow and not gpu"                 # fast unit suite (what CI runs)
pytest -m slow --timeout=3600                    # learning regressions
ruff check srl tests examples templates          # lint
black --check srl tests examples templates       # formatting
python -m sphinx -b html -W --keep-going -d docs/_build/doctrees docs docs/_build/html
```

Every YAML under `configs/` is exercised by `tests/test_configs_parse.py`, so a
new environment config is covered as soon as it is added.
