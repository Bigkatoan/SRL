"""Structured TensorBoard + terminal logger for SRL training runs."""

from __future__ import annotations

import csv
import json
import time
from collections import defaultdict, deque
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class LoggerConfig:
    """Runtime configuration for :class:`Logger`."""

    console_interval: int = 2_048
    episode_window: int = 20
    enable_tensorboard: bool = True
    enable_jsonl: bool = True
    enable_plots: bool = True
    plot_metrics: Sequence[str] | None = None
    max_console_metrics: int = 6
    console_layout: str = "multi_line"


class Logger:
    """Structured logger for TensorBoard, compact terminal output, and artifacts.

    Parameters
    ----------
    log_dir:
        Directory for TensorBoard event files and exported training artifacts.
    verbose:
        Print progress summaries to stdout.
    config:
        Optional logger runtime configuration.
    """

    def __init__(
        self,
        log_dir: str | Path = "runs",
        verbose: bool = True,
        config: LoggerConfig | None = None,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self.config = config or LoggerConfig()
        self._writer = None
        self._step = 0
        self._start_time = time.perf_counter()
        self._history: dict[str, list[tuple[int, float]]] = defaultdict(list)
        self._episodes: list[dict[str, float]] = []
        self._rolling: dict[str, deque[float]] = {
            "score": deque(maxlen=self.config.episode_window),
            "episode_length": deque(maxlen=self.config.episode_window),
        }
        self._episode_returns: list[float] = []
        self._episode_lengths: list[int] = []
        self._last_console_step = -1
        self._last_metrics: dict[str, float] = {}
        self._metadata: dict[str, Any] = {}
        self._closed = False
        self._jsonl_path = self.log_dir / "metrics.jsonl"
        self._summary_path = self.log_dir / "summary.json"
        self._history_csv_path = self.log_dir / "history.csv"

    def _get_writer(self):
        if self._writer is None:
            if not self.config.enable_tensorboard:
                return None
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._writer = SummaryWriter(log_dir=str(self.log_dir))
            except ImportError:
                pass
        return self._writer

    def set_metadata(self, **metadata: Any) -> None:
        self._metadata.update(metadata)

    def configure_env(self, num_envs: int) -> None:
        env_count = max(int(num_envs), 1)
        self._episode_returns = [0.0 for _ in range(env_count)]
        self._episode_lengths = [0 for _ in range(env_count)]

    def log(
        self,
        tag: str,
        value: float,
        step: int | None = None,
        *,
        emit_console: bool | None = None,
    ) -> None:
        step = step if step is not None else self._step
        value_f = float(value)
        writer = self._get_writer()
        if writer is not None:
            writer.add_scalar(tag, value_f, global_step=step)
        self._history[tag].append((int(step), value_f))
        self._write_metric_event(tag, value_f, int(step))
        should_print = self.verbose if emit_console is None else emit_console
        if should_print:
            print(f"[step {step:>8d}] {tag}: {value_f:.4f}", flush=True)

    def log_dict(
        self,
        metrics: dict[str, float],
        step: int | None = None,
        *,
        emit_console: bool | None = None,
    ) -> None:
        for key, value in metrics.items():
            self.log(key, value, step=step, emit_console=emit_console)

    def record_metrics(
        self,
        metrics: dict[str, float],
        *,
        step: int,
        total_steps: int | None = None,
        prefix: str | None = None,
        console: bool = True,
    ) -> None:
        if not metrics:
            return

        resolved_prefix = prefix if prefix is not None else self._metadata.get("algorithm")
        namespaced: dict[str, float] = {}
        for key, value in metrics.items():
            tag = key if "/" in key or not resolved_prefix else f"{resolved_prefix}/{key}"
            namespaced[tag] = float(value)

        self.log_dict(namespaced, step=step, emit_console=False)
        self._last_metrics.update(namespaced)

        elapsed = max(time.perf_counter() - self._start_time, 1e-6)
        progress_metrics = {
            "train/fps": float(step) / elapsed,
            "train/wall_time_sec": elapsed,
        }
        if total_steps is not None and total_steps > 0:
            progress_metrics["train/progress"] = float(step) / float(total_steps)
        self.log_dict(progress_metrics, step=step, emit_console=False)
        self._last_metrics.update(progress_metrics)

        if console:
            self._emit_progress(step=step, total_steps=total_steps, metrics=namespaced)

    def update_episodes(
        self,
        reward: Any,
        done: Any,
        truncated: Any = None,
        *,
        step: int,
        info: Any = None,
    ) -> None:
        rewards = np.asarray(reward, dtype=np.float32).reshape(-1)
        dones = np.asarray(done, dtype=bool).reshape(-1)
        truncs = (
            np.zeros_like(dones)
            if truncated is None
            else np.asarray(truncated, dtype=bool).reshape(-1)
        )

        if not self._episode_returns or len(self._episode_returns) != len(rewards):
            self.configure_env(len(rewards))

        for index, reward_i in enumerate(rewards):
            self._episode_returns[index] += float(reward_i)
            self._episode_lengths[index] += 1
            if dones[index] or truncs[index]:
                extras = self._extract_episode_extras(info, index)
                self.record_episode(
                    step=step,
                    score=self._episode_returns[index],
                    length=self._episode_lengths[index],
                    extra=extras,
                )
                self._episode_returns[index] = 0.0
                self._episode_lengths[index] = 0

    def record_episode(
        self,
        *,
        step: int,
        score: float,
        length: int,
        extra: dict[str, float] | None = None,
    ) -> None:
        score_f = float(score)
        length_f = float(length)
        self._episodes.append({"step": int(step), "score": score_f, "episode_length": length_f})
        self._rolling["score"].append(score_f)
        self._rolling["episode_length"].append(length_f)

        payload = {
            "train/score": score_f,
            "train/episode_length": length_f,
            "train/score_mean": self._rolling_mean("score"),
            "train/score_max": max(self._rolling["score"]),
            "train/episode_length_mean": self._rolling_mean("episode_length"),
            "train/episodes_completed": float(len(self._episodes)),
        }
        if extra:
            for key, value in extra.items():
                extra_key = str(key)
                self._rolling.setdefault(
                    extra_key, deque(maxlen=self.config.episode_window)
                ).append(float(value))
                payload[f"train/{extra_key}"] = float(value)
                payload[f"train/{extra_key}_mean"] = self._rolling_mean(extra_key)

        self.log_dict(payload, step=step, emit_console=False)
        self._last_metrics.update(payload)

    def finalize(self, status: str = "completed") -> None:
        if self._closed:
            return
        self._closed = True

        summary = self._build_summary(status)
        self._summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        self._export_history_csv()
        self._export_plots()

        writer = self._get_writer()
        if writer is not None:
            writer.flush()

    def set_step(self, step: int) -> None:
        self._step = step

    def close(self) -> None:
        self.finalize()
        if self._writer is not None:
            self._writer.close()

    def _write_metric_event(self, tag: str, value: float, step: int) -> None:
        if not self.config.enable_jsonl:
            return
        event = {
            "tag": tag,
            "value": value,
            "step": step,
            "time": time.time(),
        }
        with self._jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")

    def _rolling_mean(self, key: str) -> float:
        values = self._rolling.get(key)
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    def _emit_progress(
        self,
        *,
        step: int,
        total_steps: int | None,
        metrics: dict[str, float],
    ) -> None:
        if not self.verbose:
            return
        if (
            self._last_console_step >= 0
            and step - self._last_console_step < self.config.console_interval
        ):
            return

        progress_str = f"{step}"
        if total_steps is not None and total_steps > 0:
            progress = 100.0 * float(step) / float(total_steps)
            progress_str = f"{step}/{total_steps} ({progress:5.1f}%)"

        summary_pairs = [
            ("fps", self._last_metrics.get("train/fps")),
            ("score", self._last_metrics.get("train/score")),
            (f"score@{self.config.episode_window}", self._last_metrics.get("train/score_mean")),
            ("len", self._last_metrics.get("train/episode_length_mean")),
        ]

        seen = {"train/fps", "train/score", "train/score_mean", "train/episode_length_mean"}
        for key, value in metrics.items():
            if key in seen or key.endswith("_weight"):
                continue
            summary_pairs.append((key, value))
            if (
                self.config.console_layout == "single_line"
                and len(summary_pairs) >= self.config.max_console_metrics
            ):
                break

        visible_pairs = [(name, value) for name, value in summary_pairs if value is not None]
        if self.config.console_layout == "single_line":
            formatted = " | ".join(
                f"{name}={self._format_value(value)}" for name, value in visible_pairs
            )
            print(f"[train] step {progress_str} | {formatted}", flush=True)
        else:
            score_keys = {"fps", "score", f"score@{self.config.episode_window}", "len"}
            score_pairs = [(name, value) for name, value in visible_pairs if name in score_keys]
            metric_pairs = [
                (name, value) for name, value in visible_pairs if name not in score_keys
            ]
            print(f"[train] step {progress_str}", flush=True)
            if score_pairs:
                print("  rollout", flush=True)
                for name, value in score_pairs:
                    print(f"    {name}: {self._format_value(value)}", flush=True)
            if metric_pairs:
                print("  metrics", flush=True)
                for name, value in metric_pairs:
                    print(f"    {name}: {self._format_value(value)}", flush=True)
        self._last_console_step = step

    def _format_value(self, value: float) -> str:
        magnitude = abs(float(value))
        if magnitude == 0.0:
            return "0"
        if magnitude >= 1_000:
            return f"{value:.1f}"
        if magnitude >= 10:
            return f"{value:.2f}"
        return f"{value:.4f}"

    def _extract_episode_extras(self, info: Any, index: int) -> dict[str, float]:
        if info is None:
            return {}

        if isinstance(info, list):
            info_item = info[index] if index < len(info) and isinstance(info[index], dict) else {}
        elif isinstance(info, dict):
            info_item = {}
            for key, value in info.items():
                try:
                    arr = np.asarray(value)
                    if arr.ndim > 0 and index < len(arr):
                        extracted = arr[index]
                    else:
                        extracted = value
                except Exception:
                    extracted = value
                info_item[key] = extracted
        else:
            info_item = {}

        extras: dict[str, float] = {}
        for key in ("is_success", "success", "score"):
            if key not in info_item:
                continue
            try:
                extras[key] = float(np.asarray(info_item[key]).reshape(-1)[0])
            except Exception:
                continue
        return extras

    def _build_summary(self, status: str) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "status": status,
            "log_dir": str(self.log_dir),
            "metadata": self._metadata,
            "last_step": self._step,
            "last_metrics": self._last_metrics,
            "episodes_completed": len(self._episodes),
            "elapsed_sec": max(time.perf_counter() - self._start_time, 0.0),
        }
        if self._episodes:
            scores = [episode["score"] for episode in self._episodes]
            lengths = [episode["episode_length"] for episode in self._episodes]
            summary["best_score"] = max(scores)
            summary["mean_score"] = float(sum(scores) / len(scores))
            summary["mean_episode_length"] = float(sum(lengths) / len(lengths))
        return summary

    def _export_history_csv(self) -> None:
        with self._history_csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["tag", "step", "value"])
            for tag, points in sorted(self._history.items()):
                for step, value in points:
                    writer.writerow([tag, step, value])

    def _export_plots(self) -> None:
        if not self.config.enable_plots or not self._history:
            return
        plot_tags = list(self.config.plot_metrics or ())
        if not plot_tags:
            preferred = [
                "train/score",
                "train/score_mean",
                "train/episode_length_mean",
            ]
            algo_prefix = str(self._metadata.get("algorithm", ""))
            preferred.extend(
                tag for tag in self._history if algo_prefix and tag.startswith(f"{algo_prefix}/")
            )
            plot_tags = [tag for tag in preferred if tag in self._history]
            if not plot_tags:
                plot_tags = list(self._history.keys())[: min(6, len(self._history))]

        if not plot_tags:
            return

        self._export_svg_plots(plot_tags)

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return

        fig, axes = plt.subplots(len(plot_tags), 1, figsize=(10, 3 * len(plot_tags)), squeeze=False)
        for axis, tag in zip(axes.flatten(), plot_tags, strict=True):
            points = self._history.get(tag)
            if not points:
                continue
            steps, values = zip(*points, strict=True)
            axis.plot(steps, values, linewidth=1.8)
            axis.set_title(tag)
            axis.set_xlabel("step")
            axis.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(self.log_dir / "training_curves.png", dpi=160)
        plt.close(fig)

    def _export_svg_plots(self, plot_tags: list[str]) -> None:
        width = 960
        row_height = 220
        margin_left = 64
        margin_right = 24
        margin_top = 28
        plot_width = width - margin_left - margin_right
        svg_height = row_height * len(plot_tags)
        colors = ["#0f766e", "#1d4ed8", "#b45309", "#b91c1c", "#6d28d9", "#15803d"]

        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{svg_height}" viewBox="0 0 {width} {svg_height}">',  # noqa: E501
            '<rect width="100%" height="100%" fill="#f8fafc"/>',
            "<style>text{font-family: monospace; fill:#0f172a} .grid{stroke:#cbd5e1; stroke-width:1} .axis{stroke:#475569; stroke-width:1.2; fill:none}</style>",  # noqa: E501
        ]

        for row, tag in enumerate(plot_tags):
            points = self._history.get(tag)
            if not points:
                continue

            top = row * row_height
            chart_top = top + margin_top
            chart_bottom = top + row_height - 40
            chart_height = max(chart_bottom - chart_top, 1)
            values = [value for _, value in points]
            steps = [step for step, _ in points]
            min_step = min(steps)
            max_step = max(steps)
            min_value = min(values)
            max_value = max(values)
            if max_step == min_step:
                max_step += 1
            if max_value == min_value:
                max_value += 1.0

            parts.append(
                f'<text x="{margin_left}" y="{top + 18}" font-size="14" font-weight="700">{tag}</text>'  # noqa: E501
            )
            for frac in (0.0, 0.5, 1.0):
                y = chart_bottom - frac * chart_height
                value = min_value + frac * (max_value - min_value)
                parts.append(
                    f'<line class="grid" x1="{margin_left}" y1="{y:.2f}" x2="{margin_left + plot_width}" y2="{y:.2f}"/>'  # noqa: E501
                )
                parts.append(f'<text x="8" y="{y + 4:.2f}" font-size="11">{value:.3f}</text>')

            coords = []
            for step, value in points:
                x = margin_left + (step - min_step) / (max_step - min_step) * plot_width
                y = chart_bottom - (value - min_value) / (max_value - min_value) * chart_height
                coords.append(f"{x:.2f},{y:.2f}")
            coords_str = " ".join(coords)

            parts.append(
                f'<path class="axis" d="M {margin_left} {chart_top} V {chart_bottom} H {margin_left + plot_width}"/>'  # noqa: E501
            )
            parts.append(
                f'<polyline fill="none" stroke="{colors[row % len(colors)]}" stroke-width="2.5" points="{coords_str}"/>'  # noqa: E501
            )
            parts.append(
                f'<text x="{margin_left}" y="{chart_bottom + 20}" font-size="11">step {min_step}</text>'  # noqa: E501
            )
            parts.append(
                f'<text x="{margin_left + plot_width - 70}" y="{chart_bottom + 20}" font-size="11">step {max_step}</text>'  # noqa: E501
            )

        parts.append("</svg>")
        (self.log_dir / "training_curves.svg").write_text("\n".join(parts), encoding="utf-8")
