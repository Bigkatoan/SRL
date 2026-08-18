from srl.cli.benchmark import _build_parser as build_benchmark_parser
from srl.cli.train import _build_parser as build_train_parser
from srl.cli.train import _normalize_env_name
from srl.cli.visualize import _build_parser as build_visualize_parser


def test_train_parser_accepts_minimal_args() -> None:
    args = build_train_parser().parse_args(["--config", "cfg.yaml"])
    assert args.config == "cfg.yaml"
    # None at the raw-parser level, not "auto" -- main() falls back to
    # "auto" (or the YAML train.device, if set) only once it knows whether
    # --device was actually passed on the CLI. See main()'s own comment on
    # why --device/--seed/--log-interval/--eval-freq/--eval-episodes all
    # need a None default here rather than their real fallback value
    # baked in directly.
    assert args.device is None


def test_benchmark_parser_accepts_required_args() -> None:
    args = build_benchmark_parser().parse_args(["--config", "cfg.yaml", "--env", "Pendulum-v1"])
    assert args.env == "Pendulum-v1"


def test_visualize_parser_accepts_required_args() -> None:
    args = build_visualize_parser().parse_args(["--config", "cfg.yaml"])
    assert args.output_dir == "runs/pipelines"


def test_normalize_env_name_for_isaaclab() -> None:
    assert _normalize_env_name("Isaac-Ant-v0", "isaaclab") == "isaaclab:Isaac-Ant-v0"
