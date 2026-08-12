"""CLI for exporting SRL model and training pipelines without running training."""

from __future__ import annotations

import argparse
import os

from srl.cli.train import _resolve_env_name, _resolve_pipeline_outputs, _train_section
from srl.utils.pipeline_graph import render_pipeline_bundle


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="srl-visualize",
        description="Export model and training pipeline PNGs from a YAML config.",
    )
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--env", default=None, help="Optional env override")
    parser.add_argument("--algo", default=None, help="Optional algorithm override")
    parser.add_argument(
        "--output-dir", default="runs/pipelines", help="Base directory for rendered PNGs"
    )
    parser.add_argument("--model-output", default=None, help="Explicit model pipeline PNG path")
    parser.add_argument(
        "--training-output", default=None, help="Explicit training pipeline PNG path"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    _train_cfg, raw_cfg = _train_section(args.config)
    algo_name = args.algo or raw_cfg.get("algo") or os.path.basename(args.config).split("_")[0]
    env_name = _resolve_env_name(args.env, raw_cfg)
    model_output, training_output = _resolve_pipeline_outputs(
        raw_cfg,
        run_name=f"{algo_name}_{os.path.splitext(os.path.basename(args.config))[0]}",
        logdir=args.output_dir,
        cli_model_path=args.model_output or "auto",
        cli_training_path=args.training_output or "auto",
        export_only=True,
    )
    outputs = render_pipeline_bundle(
        raw_cfg,
        config_path=args.config,
        algo_name=algo_name,
        env_name=env_name,
        model_output_path=model_output,
        training_output_path=training_output,
    )
    for name, path in outputs.items():
        print(f"[srl-visualize] {name} pipeline: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
