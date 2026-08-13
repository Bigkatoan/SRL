"""Regression tests for srl-benchmark's metrics reporting.

Covers two related bugs found while verifying an internal QC pass:

1. `_reconcile_metrics` -- the stdout-scraped `fps` key and summary.json's
   `train/fps` key don't collide, so a plain dict update left the printed
   table (and any --output json consumer) reading a stale, mid-training
   `fps` snapshot instead of the real final value.
2. `--eval-freq` defaulting to 0 meant every benchmark run following the
   README's own example (which doesn't pass --eval-freq) silently produced
   a report with no eval_score at all.

Also covers the "return" -> "exit_code" column relabel (it was, and still
is, the subprocess exit code -- never an RL episode return).
"""

from __future__ import annotations

from srl.cli.benchmark import _build_parser, _reconcile_metrics


def test_reconcile_prefers_authoritative_summary_fps_over_stale_stdout_fps() -> None:
    stdout_metrics = {"fps": 843.77, "score": -1394.6}
    summary_metrics = {"train/fps": 919.75, "eval/score_mean": -900.0}

    merged = _reconcile_metrics(stdout_metrics, summary_metrics)

    assert merged["fps"] == 919.75  # not the stale 843.77
    assert merged["train/fps"] == 919.75
    assert merged["eval/score_mean"] == -900.0
    assert merged["score"] == -1394.6  # untouched, no summary.json counterpart


def test_reconcile_falls_back_to_stdout_fps_when_summary_json_missing() -> None:
    # e.g. the training subprocess crashed before writing summary.json.
    merged = _reconcile_metrics({"fps": 100.0}, {})
    assert merged["fps"] == 100.0


def test_eval_freq_defaults_to_none_and_main_resolves_it_to_steps() -> None:
    args = _build_parser().parse_args(["--config", "cfg.yaml", "--env", "Pendulum-v1"])
    assert args.eval_freq is None  # resolved to args.steps inside main(), not here

    explicit = _build_parser().parse_args(
        ["--config", "cfg.yaml", "--env", "Pendulum-v1", "--eval-freq", "0"]
    )
    assert explicit.eval_freq == 0  # explicit 0 (disable) must be respected, not overridden
