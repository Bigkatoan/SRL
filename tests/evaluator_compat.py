"""Test helper: the old ``srl.cli.train._maybe_run_evaluation`` contract,
expressed through :class:`srl.training.Evaluator` (which replaced it), so the
eval-scheduling regression tests keep checking the same behaviour."""

from __future__ import annotations

from srl.training import evaluation as eval_module


def patch_evaluate_agent(monkeypatch, fake) -> None:
    monkeypatch.setattr(eval_module, "evaluate_agent", fake)


def maybe_run_evaluation(
    agent,
    args,
    logger,
    *,
    device: str = "cpu",
    step: int,
    next_eval_step,
    eval_env=None,
    last_eval_step=None,
    best_cm=None,
    best_tracker=None,
):
    """Returns ``(next_eval_step, eval_env, last_eval_step)`` like the old helper."""
    evaluator = eval_module.Evaluator(
        eval_env,
        env_fn=lambda: object(),
        eval_freq=int(getattr(args, "eval_freq", 0)),
        n_episodes=getattr(args, "eval_episodes", 1),
        seed=args.seed + 10_000,
        env_type=getattr(args, "env_type", "flat"),
        render=bool(getattr(args, "render", False)),
        best_cm=best_cm,
        best_tracker=best_tracker,
    )
    evaluator.next_eval_step = next_eval_step
    evaluator.last_eval_step = last_eval_step
    evaluator.maybe_evaluate(agent, step, logger=logger, total_steps=args.steps)
    return evaluator.next_eval_step, evaluator.env, evaluator.last_eval_step
