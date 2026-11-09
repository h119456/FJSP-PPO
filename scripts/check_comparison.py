"""Check training settings and dataset separation for an MLP/GNN comparison."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import pickle
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppo_rs import load_instance, resolve_instance_paths  # noqa: E402
from ppo_rs.algorithms import PPOConfig  # noqa: E402
from ppo_rs.cli_utils import load_run_config  # noqa: E402
from ppo_rs.environment import ResidualStressFJSPEnv  # noqa: E402
from ppo_rs.models import build_actor_critic  # noqa: E402
from ppo_rs.train import _validate_transfer_schema  # noqa: E402


ARCHITECTURE_KEYS = {"model_type", "gnn_layers", "attention_heads", "output_dir", "initialize_from",
                     "physical_gating", "physical_gate_initial"}


def _initialization_summary(config: dict[str, Any], model: torch.nn.Module,
                            env: ResidualStressFJSPEnv) -> dict[str, Any] | None:
    source = config.get("initialize_from")
    if not source:
        return None
    path = Path(source).expanduser().resolve()
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("model_type", "mlp") != config["model_type"]:
        raise ValueError("initial checkpoint model type does not match the training config")
    for key, expected in (("observation_dim", env.OBSERVATION_DIM),
                          ("action_feature_dim", env.ACTION_FEATURE_DIM),
                          ("hidden_dim", config["hidden_dim"])):
        if checkpoint[key] != expected:
            raise ValueError(f"initial checkpoint {key} does not match the training config")
    _validate_transfer_schema(checkpoint, model, config["observation_profile"],
                              config["action_feature_profile"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_episode": int(checkpoint["episode"]),
        "source_model_type": checkpoint.get("model_type", "mlp"),
        "compatible": True,
        "optimizer_reset": True,
    }


def effective_config(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve the defaults used by train.py, including dependent defaults."""
    result = {
        **asdict(PPOConfig()),
        "seed": 7,
        "episodes": 40,
        "model_type": "mlp",
        "observation_profile": ResidualStressFJSPEnv.DEFAULT_OBSERVATION_PROFILE,
        "action_feature_profile": "compact21",
        "hidden_dim": 128,
        "gnn_layers": 2,
        "attention_heads": 4,
        "episodes_per_update": 1,
        "checkpoint_interval": 50,
        "enable_plots": True,
        "device": "auto",
        "output_dir": "outputs/default_run",
        "output_dir_mode": "existing",
        "distributed_world_size": 1,
        **config,
    }
    result.setdefault("eval_seed", result["seed"] + 1000)
    result.setdefault("final_learning_rate", result["learning_rate"])
    result.setdefault("final_entropy_coef", result["entropy_coef"])
    result.setdefault("validation_interval", max(1, result["episodes_per_update"]))
    result.setdefault("latest_checkpoint_interval", result["episodes_per_update"])
    result["train_instances"] = config.get("train_instances", config.get("train_instance"))
    result["eval_instances"] = config.get(
        "eval_instances", config.get("eval_instance", result["train_instances"])
    )
    result["test_instances"] = config.get("test_instances", config.get("test_instance"))
    for key in ("train_instance", "eval_instance", "test_instance"):
        result.pop(key, None)
    return result


def _constraints(config: dict[str, Any]) -> list[str]:
    errors = []
    positive_ints = (
        "episodes", "episodes_per_update", "validation_interval",
        "latest_checkpoint_interval", "checkpoint_interval", "mini_batch_size",
        "ppo_epochs", "hidden_dim", "attention_heads", "distributed_world_size",
    )
    for key in positive_ints:
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            errors.append(f"{key} must be a positive integer")
    if not errors:
        if config["episodes"] < config["episodes_per_update"]:
            errors.append("episodes must be at least episodes_per_update")
        if config["mini_batch_size"] % config["distributed_world_size"]:
            errors.append("global mini_batch_size must be divisible by distributed_world_size")
        for key in ("validation_interval", "latest_checkpoint_interval", "checkpoint_interval"):
            if config[key] % config["episodes_per_update"]:
                errors.append(f"{key} must be divisible by episodes_per_update")
        if config["model_type"] == "gnn_attention":
            if config["gnn_layers"] != 2:
                errors.append("gnn_layers must be 2")
            if config["hidden_dim"] % config["attention_heads"]:
                errors.append("hidden_dim must be divisible by attention_heads")
    for key in ("learning_rate", "final_learning_rate", "max_grad_norm", "clip_ratio"):
        value = config[key]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            errors.append(f"{key} must be finite and positive")
    if isinstance(config["clip_ratio"], (int, float)) and config["clip_ratio"] >= 1:
        errors.append("clip_ratio must be less than 1")
    for key in ("value_coef", "entropy_coef", "final_entropy_coef"):
        value = config[key]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            errors.append(f"{key} must be finite and non-negative")
    for key in ("gamma", "gae_lambda"):
        value = config[key]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            errors.append(f"{key} must be between 0 and 1")
    if config["output_dir_mode"] not in {"unique", "existing"}:
        errors.append("output_dir_mode must be unique or existing")
    return errors


def _dataset_summary(config: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    errors = []
    summaries = {}
    split_hashes: dict[str, set[str]] = {}
    for split, key in (("train", "train_instances"), ("validation", "eval_instances"), ("test", "test_instances")):
        source = config[key]
        if source is None:
            errors.append(f"{key} is required for a train/validation/test comparison")
            continue
        paths = resolve_instance_paths(source)
        hashes = []
        for path in paths:
            # Match supported instance formats and ignore display names when
            # checking that a renamed copy has not crossed dataset boundaries.
            content = load_run_config(path)
            content.pop("name", None)
            canonical = json.dumps(content, sort_keys=True, separators=(",", ":"))
            hashes.append(hashlib.sha256(canonical.encode()).hexdigest())
        split_hashes[split] = set(hashes)
        summaries[split] = {
            "source": source,
            "instance_count": len(paths),
            "unique_content_count": len(set(hashes)),
            "content_sha256": hashlib.sha256("\n".join(sorted(hashes)).encode()).hexdigest(),
        }
    for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
        duplicates = split_hashes.get(first, set()) & split_hashes.get(second, set())
        if duplicates:
            errors.append(f"dataset leakage: {first}/{second} share {len(duplicates)} instance contents (ignoring name)")
    return summaries, errors


def _model_summary(config: dict[str, Any]) -> dict[str, Any]:
    first = resolve_instance_paths(config["train_instances"])[0]
    env = ResidualStressFJSPEnv(
        load_instance(first), observation_profile=config["observation_profile"],
        action_feature_profile=config["action_feature_profile"],
    )
    observation, info = env.reset(seed=config["seed"])
    graph = env.get_graph_state(info["valid_actions"])
    # Inspect shapes without consuming the caller's model initialization RNG.
    with torch.random.fork_rng(devices=[]):
        model = build_actor_critic(
            config["model_type"], env.OBSERVATION_DIM, env.ACTION_FEATURE_DIM,
            config["hidden_dim"], node_feature_dim=env.GRAPH_NODE_FEATURE_DIM,
            edge_feature_dim=env.GRAPH_EDGE_FEATURE_DIM,
            node_type_count=env.GRAPH_NODE_TYPE_COUNT,
            edge_type_count=env.GRAPH_EDGE_TYPE_COUNT,
            gnn_layers=config["gnn_layers"], attention_heads=config["attention_heads"],
            physical_gating=config.get("physical_gating", False),
            physical_gate_initial=config.get("physical_gate_initial", 0.9),
        )
        initialization = _initialization_summary(config, model, env)
    if len(observation) != env.OBSERVATION_DIM or any(
        len(row) != env.ACTION_FEATURE_DIM for row in info["action_features"]
    ):
        raise ValueError("runtime state/action shape differs from declared dimensions")
    return {
        "environment_semantics_version": env.ENVIRONMENT_SEMANTICS_VERSION,
        "model_type": config["model_type"],
        "observation_profile": config["observation_profile"],
        "observation_dim": len(observation),
        "observation_names": list(env.OBSERVATION_NAMES),
        "action_feature_profile": config["action_feature_profile"],
        "action_feature_dim": env.ACTION_FEATURE_DIM,
        "action_feature_names": list(env.ACTION_FEATURE_NAMES),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "requires_graph": bool(model.requires_graph),
        "graph_node_feature_dim": int(graph["node_features"].shape[1]) if model.requires_graph else None,
        "graph_edge_feature_dim": int(graph["edge_features"].shape[1]) if model.requires_graph else None,
        "graph_node_type_count": env.GRAPH_NODE_TYPE_COUNT if model.requires_graph else None,
        "graph_edge_type_count": env.GRAPH_EDGE_TYPE_COUNT if model.requires_graph else None,
        "reserved_dynamic_state_dim": sum(name in env.OBSERVATION_NAMES for name in env.DYNAMIC_OBSERVATION_NAMES),
        "reserved_dynamic_node_dim": (
            env.GRAPH_NODE_FEATURE_DIM - env.GRAPH_BASE_NODE_FEATURE_DIM
            if model.requires_graph else 0
        ),
        "planned_rollout_updates": math.ceil(config["episodes"] / config["episodes_per_update"]),
        "distributed_world_size": config["distributed_world_size"],
        "global_mini_batch_size": config["mini_batch_size"],
        "local_mini_batch_size": config["mini_batch_size"] // config["distributed_world_size"],
        "initialization": initialization,
    }


def audit_comparison(mlp_config: dict[str, Any], gnn_config: dict[str, Any]) -> dict[str, Any]:
    configs = {"mlp": effective_config(mlp_config), "gnn": effective_config(gnn_config)}
    errors = []
    if configs["mlp"]["model_type"] != "mlp":
        errors.append("MLP config must use model_type=mlp")
    if configs["gnn"]["model_type"] != "gnn_attention":
        errors.append("GNN config must use model_type=gnn_attention")
    if bool(configs["mlp"].get("initialize_from")) != bool(configs["gnn"].get("initialize_from")):
        errors.append("both models must initialize from checkpoints, or both train from scratch")
    unique_runs = all(config["output_dir_mode"] == "unique" for config in configs.values())
    if not unique_runs and Path(configs["mlp"]["output_dir"]).resolve() == Path(configs["gnn"]["output_dir"]).resolve():
        errors.append("MLP and GNN output_dir must differ")
    differences = {
        key: {label: config.get(key) for label, config in configs.items()}
        for key in sorted(set(configs["mlp"]) | set(configs["gnn"]))
        if key not in ARCHITECTURE_KEYS and configs["mlp"].get(key) != configs["gnn"].get(key)
    }
    if differences:
        errors.append("training settings differ: " + ", ".join(differences))
    models = {}
    datasets = {}
    for label, config in configs.items():
        constraint_errors = _constraints(config)
        errors.extend(f"{label}: {error}" for error in constraint_errors)
        try:
            datasets[label], dataset_errors = _dataset_summary(config)
            errors.extend(f"{label}: {error}" for error in dataset_errors)
            if not constraint_errors:
                models[label] = _model_summary(config)
        except (OSError, ValueError, TypeError, KeyError, RuntimeError, EOFError, pickle.UnpicklingError) as exc:
            errors.append(f"{label}: {exc}")
    return {
        "passed": not errors,
        "scope": "Existing complete MLP and GNN methods under matched training settings",
        "errors": errors,
        "setting_differences": differences,
        "allowed_architecture_or_output_differences": sorted(ARCHITECTURE_KEYS),
        "effective_configs": configs,
        "models": models,
        "datasets": datasets,
        "limitations": [
            "Matched global state/action features do not imply identical total input: GNN also reads the graph.",
            "Equal hidden_dim does not imply equal parameter count or computation.",
            "Equal episodes does not imply identical environment decisions or Adam steps; report actual steps and time.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlp-config", required=True)
    parser.add_argument("--gnn-config", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    try:
        result = audit_comparison(load_run_config(args.mlp_config), load_run_config(args.gnn_config))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        result = {"passed": False, "errors": [str(exc)]}
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
