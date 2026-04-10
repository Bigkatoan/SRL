# Contributing

Thank you for your interest in contributing to SRL!

---

## Development setup

```bash
git clone https://github.com/Bigkatoan/SRL.git
cd SRL
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
```

---

## Running tests

```bash
# Unit and integration tests (main venv, Python 3.10)
pytest tests/ -v

# Environment smoke tests (tests/venv, Python 3.11 with IsaacLab)
source tests/venv/bin/activate
python tests/test_environments.py
```

---

## Code style

- Black formatting (`black srl/`)
- Type hints on all public functions
- Google-style docstrings

---

## Adding a new environment

1. Check if it needs a custom wrapper (Goal → `GoalEnvWrapper`, Isaac → `IsaacLabWrapper`)
2. Create `configs/envs/<env>_<algo>.yaml`
3. Add a training script (or extend an existing one) in `examples/envs/`
4. Add a smoke test entry in `tests/test_environments.py`
5. Add a docs page in `docs/environments/`

---

## Adding a new algorithm

1. Create `srl/algorithms/<name>.py` extending `BaseAgent`
2. Add a `<Name>Config` dataclass in `srl/core/config.py`
3. Register losses in `srl/losses/rl_losses.py`
4. Add a training script in `examples/`
5. Document in `docs/algorithms.md`

---

## Pull request checklist

- [ ] Tests pass (`pytest tests/ -v`)
- [ ] Env smoke tests pass
- [ ] Type hints added
- [ ] Docstrings updated
- [ ] `docs/` updated if public API changed
