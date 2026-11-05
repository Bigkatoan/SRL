# Benchmarks

```{note}
No finished benchmark runs yet: SRL columns read *pending*. Launch the
matrix with `scripts/benchmark/launch_phase1.sh`, then regenerate this
page with `python scripts/benchmark/report.py`.
```

SRL's SAC, TD3 and PPO on three Gymnasium MuJoCo tasks, trained with the
[rl-baselines3-zoo](https://github.com/DLR-RM/rl-baselines3-zoo) hyperparameters
(configs in `configs/benchmarks/mujoco/`) and compared with the published
Stable-Baselines3 zoo and CleanRL numbers.

## Protocol

- Environments: `HalfCheetah-v5`, `Hopper-v5`, `Walker2d-v5`, a single
  environment per run (`n_envs: 1`), 1M environment steps.
- Every 10k steps: 10 deterministic episodes on
  a separate evaluation environment.
- **Final metric** of a run: mean of its last 5 evaluations.
- Across seeds: mean ± sample std, and the interquartile mean (IQM) with a
  95% percentile-bootstrap CI (2000 resamples, rliable if installed, else built-in implementation). With
  3 seeds the IQM equals the mean and the CI is wide: read it as a range.
- *SRL / zoo*: SRL mean divided by the SB3 zoo number; the aggregate table
  pools these ratios over environments (stratified bootstrap).

## Results

| Env | Algo | Seeds | Steps | SRL mean ± std | SRL IQM [95% CI] | SB3 zoo | CleanRL | SRL / zoo |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| HalfCheetah-v5 | SAC | 0 | – | pending | pending | 9,535 ± 100 | 9,635 ± 1,424 | – |
| HalfCheetah-v5 | TD3 | 0 | – | pending | pending | 9,656 ± 970 | 9,583 ± 126 | – |
| HalfCheetah-v5 | PPO | 0 | – | pending | pending | 5,819 ± 664 | 1,443 ± 46 | – |
| Hopper-v5 | SAC | 0 | – | pending | pending | 2,326 ± 1,130 | 2,310 ± 343 | – |
| Hopper-v5 | TD3 | 0 | – | pending | pending | 3,606 ± 4 | 3,135 ± 360 | – |
| Hopper-v5 | PPO | 0 | – | pending | pending | 2,410 ± 10 | 2,383 ± 272 | – |
| Walker2d-v5 | SAC | 0 | – | pending | pending | 3,863 ± 254 | 3,591 ± 911 | – |
| Walker2d-v5 | TD3 | 0 | – | pending | pending | 4,718 ± 46 | 4,058 ± 659 | – |
| Walker2d-v5 | PPO | 0 | – | pending | pending | 3,479 ± 822 | 2,288 ± 572 | – |

### Aggregate (normalised by SB3 zoo)

| Algo | Envs | Runs | IQM of SRL / SB3-zoo [95% CI] |
|---|---:|---:|---:|
| SAC | 0 | 0 | pending |
| TD3 | 0 | 0 | pending |
| PPO | 0 | 0 | pending |

## Reference numbers

The reference protocols differ from ours, so treat them as context and not as
exact targets:

- **SB3 rl-zoo** ([source](https://github.com/DLR-RM/rl-baselines3-zoo/blob/61956d90d9f8ac126043ed4024efa12a462d97e1/benchmark.md)): single trained agent per row, 1M steps, deterministic eval over ~150k steps; std = across eval episodes; envs are -v3.
- **CleanRL** ([sac](https://github.com/vwxyzjn/cleanrl/blob/fe8d8a03c41a7ef5b523e2e354bd01c363e786bb/docs/rl-algorithms/sac.md), [td3](https://github.com/vwxyzjn/cleanrl/blob/fe8d8a03c41a7ef5b523e2e354bd01c363e786bb/docs/rl-algorithms/td3.md), [ppo](https://github.com/vwxyzjn/cleanrl/blob/fe8d8a03c41a7ef5b523e2e354bd01c363e786bb/docs/benchmark/ppo_continuous_action.md)): mean ± std over 3 seeds of training episodic return at the end of training (1M steps); SAC on -v2 envs, TD3/PPO on -v4 envs.

CleanRL's PPO uses its own default hyperparameters, not the tuned rl-zoo
values our PPO configs use. All values live in
`scripts/benchmark/reference.yaml`, with the date they were checked.

## Reproducing

```bash
scripts/benchmark/launch_phase1.sh            # start, detached
python scripts/benchmark/run_matrix.py status # progress
python scripts/benchmark/report.py            # regenerate this page
```
