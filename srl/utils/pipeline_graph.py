"""Render model and training pipelines from YAML configs into PNG graphs."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

_KIND_COLORS = {
    "config": "#F4F1DE",
    "encoder": "#DDECF7",
    "actor": "#D8F3DC",
    "critic": "#FDE2E4",
    "loss": "#FFF1C1",
    "env": "#E9D8FD",
    "buffer": "#FCE1A8",
    "update": "#D0F4DE",
    "artifact": "#CCE3DE",
    "process": "#E7ECEF",
}


def render_pipeline_bundle(
    raw_cfg: dict[str, Any],
    *,
    config_path: str,
    algo_name: str,
    env_name: str,
    model_output_path: str | Path | None = None,
    training_output_path: str | Path | None = None,
) -> dict[str, str]:
    outputs: dict[str, str] = {}
    if model_output_path:
        render_model_pipeline(raw_cfg, config_path=config_path, output_path=model_output_path)
        outputs["model"] = str(Path(model_output_path))
    if training_output_path:
        render_training_pipeline(
            raw_cfg,
            config_path=config_path,
            algo_name=algo_name,
            env_name=env_name,
            output_path=training_output_path,
        )
        outputs["training"] = str(Path(training_output_path))
    return outputs


def render_model_pipeline(
    raw_cfg: dict[str, Any], *, config_path: str, output_path: str | Path
) -> None:
    nodes: list[tuple[str, str, str]] = []
    edges: list[tuple[str, str, str]] = []

    nodes.append(("yaml_config", _label("YAML Config", [Path(config_path).name]), "config"))
    for encoder in raw_cfg.get("encoders", []):
        details = [
            f"type={encoder.get('type', 'unknown')}",
            _dim_summary(encoder),
            _layers_summary(encoder.get("layers", [])),
        ]
        nodes.append((encoder["name"], _label(encoder["name"], details), "encoder"))
        edges.append(("yaml_config", encoder["name"], "defines"))

    actor = raw_cfg.get("actor")
    if actor:
        details = [
            f"type={actor.get('type', 'unknown')}",
            f"action_dim={actor.get('action_dim', 'n/a')}",
            _layers_summary(actor.get("layers", [])),
        ]
        nodes.append((actor["name"], _label(actor["name"], details), "actor"))
        edges.append(("yaml_config", actor["name"], "defines"))

    critic = raw_cfg.get("critic")
    if critic:
        details = [
            f"type={critic.get('type', 'unknown')}",
            f"action_dim={critic.get('action_dim', 'n/a')}",
            _layers_summary(critic.get("layers", [])),
        ]
        nodes.append((critic["name"], _label(critic["name"], details), "critic"))
        edges.append(("yaml_config", critic["name"], "defines"))

    for flow in raw_cfg.get("flows", []):
        if "->" not in flow:
            continue
        src, dst = [part.strip() for part in flow.split("->", 1)]
        edges.append((src, dst, "flow"))

    for loss in raw_cfg.get("losses", []):
        loss_id = f"loss_{loss['name']}"
        details = [f"weight={loss.get('weight', 1.0)}"]
        nodes.append((loss_id, _label(loss["name"], details), "loss"))
        edges.append(("yaml_config", loss_id, "defines"))
        if actor:
            edges.append((actor["name"], loss_id, "optimizes"))
        if critic:
            edges.append((critic["name"], loss_id, "optimizes"))

    dot = _build_dot(
        title=f"Model Pipeline: {Path(config_path).name}",
        nodes=nodes,
        edges=edges,
        rankdir="LR",
    )
    _render_dot_png(dot, output_path)


def render_training_pipeline(
    raw_cfg: dict[str, Any],
    *,
    config_path: str,
    algo_name: str,
    env_name: str,
    output_path: str | Path,
) -> None:
    visualization = raw_cfg.get("visualization") or {}
    custom_pipeline = visualization.get("training_pipeline") or {}
    custom_nodes = custom_pipeline.get("nodes") or []
    custom_edges = custom_pipeline.get("edges") or []

    if custom_nodes:
        nodes = [
            (
                node["id"],
                _label(node.get("label", node["id"]), node.get("details", [])),
                node.get("kind", "process"),
            )
            for node in custom_nodes
        ]
        edges = [_edge_from_any(edge) for edge in custom_edges]
    else:
        nodes, edges = _default_training_pipeline(
            raw_cfg, config_path=config_path, algo_name=algo_name, env_name=env_name
        )

    dot = _build_dot(
        title=f"Training Pipeline: {algo_name.upper()} | {env_name}",
        nodes=nodes,
        edges=edges,
        rankdir="TB",
    )
    _render_dot_png(dot, output_path)


def _default_training_pipeline(
    raw_cfg: dict[str, Any], *, config_path: str, algo_name: str, env_name: str
):
    train_cfg = raw_cfg.get("train") or {}
    off_policy = algo_name in {"sac", "ddpg", "td3"}
    nodes = [
        (
            "yaml_config",
            _label("YAML Config", [Path(config_path).name, f"algo={algo_name}"]),
            "config",
        ),
        ("build_model", _label("Build Model", ["ModelBuilder.from_yaml"]), "process"),
        (
            "create_env",
            _label("Create Environment", [env_name, f"n_envs={train_cfg.get('n_envs', 1)}"]),
            "env",
        ),
        ("collect", _label("Collect Rollouts", ["policy -> env.step"]), "process"),
        ("logger", _label("Logger / Artifacts", ["metrics", "plots", "summary"]), "artifact"),
        ("checkpoint", _label("Checkpoint", ["periodic saves"]), "artifact"),
    ]
    edges = [
        ("yaml_config", "build_model", "configure"),
        ("yaml_config", "create_env", "configure"),
        ("build_model", "collect", "policy"),
        ("create_env", "collect", "env"),
        ("collect", "logger", "metrics"),
        ("collect", "checkpoint", "weights"),
    ]

    if off_policy:
        nodes.extend(
            [
                (
                    "replay_buffer",
                    _label("Replay Buffer", [f"batch_size={train_cfg.get('batch_size', 256)}"]),
                    "buffer",
                ),
                (
                    "critic_update",
                    _label(
                        "Critic Update", [f"gradient_steps={train_cfg.get('gradient_steps', 1)}"]
                    ),
                    "update",
                ),
                ("actor_update", _label("Actor Update", ["policy gradient"]), "update"),
                (
                    "target_update",
                    _label("Target Sync", [f"tau={train_cfg.get('tau', 0.005)}"]),
                    "update",
                ),
            ]
        )
        edges.extend(
            [
                ("collect", "replay_buffer", "store"),
                ("replay_buffer", "critic_update", "sample"),
                ("critic_update", "actor_update", "q-values"),
                ("actor_update", "target_update", "weights"),
                ("critic_update", "logger", "losses"),
                ("actor_update", "logger", "losses"),
                ("target_update", "checkpoint", "weights"),
            ]
        )
    else:
        nodes.extend(
            [
                (
                    "rollout_buffer",
                    _label("Rollout Buffer", [f"n_steps={train_cfg.get('n_steps', 0)}"]),
                    "buffer",
                ),
                (
                    "compute_adv",
                    _label(
                        "Compute Returns / GAE", [f"gae_lambda={train_cfg.get('gae_lambda', 0.95)}"]
                    ),
                    "update",
                ),
                (
                    "policy_update",
                    _label("Policy / Value Update", [f"n_epochs={train_cfg.get('n_epochs', 1)}"]),
                    "update",
                ),
            ]
        )
        edges.extend(
            [
                ("collect", "rollout_buffer", "store"),
                ("rollout_buffer", "compute_adv", "returns"),
                ("compute_adv", "policy_update", "batches"),
                ("policy_update", "logger", "losses"),
                ("policy_update", "checkpoint", "weights"),
            ]
        )
    return nodes, edges


def _dim_summary(block: dict[str, Any]) -> str:
    if block.get("input_dim") is not None:
        return f"input={block['input_dim']} | latent={block.get('latent_dim', 'n/a')}"
    if block.get("input_shape") is not None:
        return f"input_shape={block['input_shape']} | latent={block.get('latent_dim', 'n/a')}"
    return f"latent={block.get('latent_dim', 'n/a')}"


def _layers_summary(layers: list[Any]) -> str:
    if not layers:
        return "layers=output_only"
    parts: list[str] = []
    for layer in layers:
        if isinstance(layer, int):
            parts.append(str(layer))
            continue
        out_dim = layer.get("out_features") or layer.get("out_channels") or "?"
        activation = layer.get("activation", "linear")
        parts.append(f"{out_dim}:{activation}")
    return "layers=" + " | ".join(parts)


def _label(title: str, details: list[str]) -> str:
    clean_details = [detail for detail in details if detail]
    return "\n".join([title] + clean_details)


def _edge_from_any(edge: Any) -> tuple[str, str, str]:
    if isinstance(edge, str):
        src, dst = [part.strip() for part in edge.split("->", 1)]
        return (src, dst, "")
    return (edge["src"], edge["dst"], edge.get("label", ""))


def _build_dot(
    *,
    title: str,
    nodes: list[tuple[str, str, str]],
    edges: list[tuple[str, str, str]],
    rankdir: str,
) -> str:
    lines = [
        "digraph Pipeline {",
        f"  graph [fontname=Helvetica, rankdir={rankdir}, bgcolor=white, labeljust=l, labelloc=t, fontsize=20];",  # noqa: E501
        '  node [shape=box, style="rounded,filled", fontname=Helvetica, fontsize=11, color="#3A506B"];',  # noqa: E501
        '  edge [fontname=Helvetica, fontsize=10, color="#5C677D"];',
        '  labelloc="t";',
        f'  label="{_escape(title)}";',
    ]
    for node_id, label, kind in nodes:
        fill = _KIND_COLORS.get(kind, _KIND_COLORS["process"])
        lines.append(f'  "{_escape(node_id)}" [label="{_escape(label)}", fillcolor="{fill}"];')
    for src, dst, label in edges:
        if label:
            lines.append(f'  "{_escape(src)}" -> "{_escape(dst)}" [label="{_escape(label)}"];')
        else:
            lines.append(f'  "{_escape(src)}" -> "{_escape(dst)}";')
    lines.append("}")
    return "\n".join(lines)


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _render_dot_png(dot_source: str, output_path: str | Path) -> None:
    dot_path = shutil.which("dot")
    if dot_path is None:
        raise RuntimeError("Graphviz 'dot' executable is required to render PNG pipeline graphs.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.run(
        [dot_path, "-Tpng", "-o", str(output_path)],
        input=dot_source,
        text=True,
        capture_output=True,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(f"Graphviz failed to render {output_path}: {process.stderr.strip()}")
