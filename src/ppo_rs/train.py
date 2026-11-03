"""Reproducible PPO training entry point."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ppo_rs import load_instances
from ppo_rs.algorithms import PPO, PPOConfig, RolloutBuffer, collect_episode
from ppo_rs.cli_utils import choose_device, load_run_config, set_seed, write_json
from ppo_rs.environment import ResidualStressFJSPEnv
from ppo_rs.evaluation import evaluate_model, plot_gantt, plot_training_curve
from ppo_rs.models import build_actor_critic


def _mean_validation_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Average scalar validation metrics while retaining per-instance results."""

    scalar_keys = {
        key
        for key in results[0]["metrics"]
        if all(
            isinstance(result["metrics"].get(key), (int, float))
            and not isinstance(result["metrics"].get(key), bool)
            for result in results
        )
    }
    aggregate = {
        key: sum(float(result["metrics"][key]) for result in results) / len(results)
        for key in sorted(scalar_keys)
    }
    aggregate.update(
        {
            "policy": "ppo",
            "instance_count": len(results),
            "terminated": all(bool(result["metrics"]["terminated"]) for result in results),
            "truncated": any(bool(result["metrics"]["truncated"]) for result in results),
            "instances": results,
        }
    )
    return aggregate


def _evaluate_validation_set(
    instances: tuple,
    model: nn.Module,
    *,
    seed: int,
    observation_profile: str,
    action_feature_profile: str,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    results: list[dict[str, Any]] = []
    schedules: dict[str, list[dict[str, Any]]] = {}
    for index, instance in enumerate(instances):
        metrics, schedule = evaluate_model(
            ResidualStressFJSPEnv(
                instance,
                observation_profile=observation_profile,
                action_feature_profile=action_feature_profile,
            ),
            model,
            seed=seed + index,
        )
        results.append({"name": instance.name, "metrics": metrics})
        schedules[instance.name] = schedule
    return _mean_validation_metrics(results), schedules


def _is_valid_evaluation(evaluation: dict[str, Any]) -> bool:
    return (
        bool(evaluation["terminated"])
        and not bool(evaluation["truncated"])
        and all(
            int(result["metrics"]["schedule_violation_count"]) == 0
            for result in evaluation["instances"]
        )
    )


def _checkpoint_payload(
    *,
    model: nn.Module,
    algorithm: PPO,
    config: dict[str, Any],
    episode: int,
    total_environment_steps: int,
    best_validation_objective: float,
    observation_dim: int,
    action_feature_dim: int,
) -> dict[str, Any]:
    """Create an evaluation-compatible checkpoint that can also resume training."""

    return {
        "environment_semantics_version": ResidualStressFJSPEnv.ENVIRONMENT_SEMANTICS_VERSION,
        "episode": episode,
        "total_environment_steps": total_environment_steps,
        "best_validation_objective": best_validation_objective,
        "model_state": model.state_dict(),
        "optimizer_state": algorithm.optimizer.state_dict(),
        "hidden_dim": int(config.get("hidden_dim", 128)),
        "observation_dim": observation_dim,
        "action_feature_dim": action_feature_dim,
        "observation_profile": str(
            config.get(
                "observation_profile",
                ResidualStressFJSPEnv.DEFAULT_OBSERVATION_PROFILE,
            )
        ),
        "action_feature_profile": str(
            config.get("action_feature_profile", "compact21")
        ),
        "model_type": str(getattr(model, "model_type", "mlp")),
        "model_kwargs": dict(getattr(model, "checkpoint_kwargs", {})),
        "config": config,
        "torch_random_state": torch.get_rng_state(),
    }


def _save_checkpoint(path: Path, **kwargs: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_checkpoint_payload(**kwargs), path)


def train(
    config: dict,
    *,
    resume_from: str | Path | None = None,
    initialize_from: str | Path | None = None,
) -> tuple[nn.Module, list[dict[str, Any]], dict[str, Any]]:
    if resume_from is not None and initialize_from is not None:
        raise ValueError("resume_from and initialize_from are mutually exclusive")
    seed = int(config.get("seed", 7))
    set_seed(seed)
    device = choose_device(str(config.get("device", "auto")))
    train_source = config.get("train_instances", config.get("train_instance"))
    if train_source is None:
        raise ValueError("config needs train_instances or train_instance")
    train_instances = load_instances(train_source)
    observation_profile = str(
        config.get(
            "observation_profile",
            ResidualStressFJSPEnv.DEFAULT_OBSERVATION_PROFILE,
        )
    )
    action_feature_profile = str(
        config.get("action_feature_profile", "compact21")
    )
    env = ResidualStressFJSPEnv(
        train_instances[0],
        observation_profile=observation_profile,
        action_feature_profile=action_feature_profile,
    )
    observation_dim = env.OBSERVATION_DIM
    action_feature_dim = env.ACTION_FEATURE_DIM
    model_type = str(config.get("model_type", "mlp"))
    model = build_actor_critic(
        model_type,
        observation_dim,
        action_feature_dim,
        int(config.get("hidden_dim", 128)),
        node_feature_dim=env.GRAPH_NODE_FEATURE_DIM,
        edge_feature_dim=env.GRAPH_EDGE_FEATURE_DIM,
        node_type_count=env.GRAPH_NODE_TYPE_COUNT,
        edge_type_count=env.GRAPH_EDGE_TYPE_COUNT,
        gnn_layers=int(config.get("gnn_layers", 2)),
        attention_heads=int(config.get("attention_heads", 4)),
    ).to(device)
    ppo_config = PPOConfig(
        **{
            key: config[key]
            for key in PPOConfig.__dataclass_fields__
            if key in config
        }
    )
    algorithm = PPO(model, ppo_config)
    initial_learning_rate = ppo_config.learning_rate
    final_learning_rate = float(
        config.get("final_learning_rate", initial_learning_rate)
    )
    initial_entropy_coef = ppo_config.entropy_coef
    final_entropy_coef = float(
        config.get("final_entropy_coef", initial_entropy_coef)
    )
    if final_learning_rate <= 0:
        raise ValueError("final_learning_rate must be positive")
    if final_entropy_coef < 0:
        raise ValueError("final_entropy_coef must be non-negative")
    eval_source = config.get(
        "eval_instances", config.get("eval_instance", train_source)
    )
    eval_instances = load_instances(eval_source)
    output = Path(config.get("output_dir", "outputs/default_run"))
    checkpoints = output / "checkpoints"
    output.mkdir(parents=True, exist_ok=True)

    episodes = int(config.get("episodes", 40))
    episodes_per_update = int(config.get("episodes_per_update", 1))
    validation_interval = int(config.get("validation_interval", max(1, episodes_per_update)))
    latest_interval = int(config.get("latest_checkpoint_interval", episodes_per_update))
    periodic_interval = int(config.get("checkpoint_interval", 50))
    for name, value in {
        "episodes": episodes,
        "episodes_per_update": episodes_per_update,
        "validation_interval": validation_interval,
        "latest_checkpoint_interval": latest_interval,
        "checkpoint_interval": periodic_interval,
    }.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")

    history: list[dict[str, Any]] = []
    validation_history: list[dict[str, Any]] = []
    order = list(range(len(train_instances)))
    sampler = random.Random(seed)
    rollout = RolloutBuffer()
    total_environment_steps = 0
    best_validation_objective = float("inf")
    start_episode = 1
    latest_evaluation: dict[str, Any] | None = None
    latest_schedules: dict[str, list[dict[str, Any]]] | None = None
    common_checkpoint_args = {
        "model": model,
        "algorithm": algorithm,
        "config": config,
        "observation_dim": observation_dim,
        "action_feature_dim": action_feature_dim,
    }

    if initialize_from is not None:
        initialization_path = Path(initialize_from)
        checkpoint = torch.load(
            initialization_path, map_location=device, weights_only=False
        )
        if int(checkpoint["observation_dim"]) != observation_dim:
            raise ValueError(
                "initial checkpoint observation dimension does not match the environment"
            )
        if int(checkpoint["action_feature_dim"]) != action_feature_dim:
            raise ValueError(
                "initial checkpoint action dimension does not match the environment"
            )
        if int(checkpoint["hidden_dim"]) != int(config.get("hidden_dim", 128)):
            raise ValueError(
                "initial checkpoint hidden dimension does not match the training config"
            )
        if str(checkpoint.get("model_type", "mlp")) != model_type:
            raise ValueError(
                "initial checkpoint model type does not match the training config"
            )
        model.load_state_dict(checkpoint["model_state"])
        # Deliberately do not restore optimizer state, episode counters, or
        # histories: dynamic fine-tuning starts a new run with fresh Adam
        # moments and its own learning-rate schedule.
        print(
            f"initialized model weights from {initialization_path} "
            f"(source_episode={int(checkpoint['episode'])}); optimizer reset"
        )

    if resume_from is not None:
        resume_path = Path(resume_from)
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        if int(checkpoint["environment_semantics_version"]) != int(
            ResidualStressFJSPEnv.ENVIRONMENT_SEMANTICS_VERSION
        ):
            raise ValueError("checkpoint environment semantics do not match the current environment")
        if int(checkpoint["observation_dim"]) != observation_dim:
            raise ValueError("checkpoint observation dimension does not match the current environment")
        if int(checkpoint["action_feature_dim"]) != action_feature_dim:
            raise ValueError("checkpoint action dimension does not match the current environment")
        if int(checkpoint["hidden_dim"]) != int(config.get("hidden_dim", 128)):
            raise ValueError("checkpoint hidden dimension does not match the training config")
        if str(checkpoint.get("model_type", "mlp")) != model_type:
            raise ValueError("checkpoint model type does not match the training config")

        model.load_state_dict(checkpoint["model_state"])
        algorithm.optimizer.load_state_dict(checkpoint["optimizer_state"])
        torch.set_rng_state(checkpoint["torch_random_state"].cpu())
        completed_episode = int(checkpoint["episode"])
        if completed_episode >= episodes:
            raise ValueError(
                f"checkpoint already reached episode {completed_episode}, "
                f"which is not below configured episodes={episodes}"
            )
        start_episode = completed_episode + 1
        total_environment_steps = int(checkpoint["total_environment_steps"])
        best_validation_objective = float(checkpoint["best_validation_objective"])

        history_path = output / "training_history.json"
        validation_history_path = output / "validation_history.json"
        if not history_path.exists() or not validation_history_path.exists():
            raise FileNotFoundError(
                "resuming requires the existing training_history.json and "
                "validation_history.json"
            )
        history = json.loads(history_path.read_text(encoding="utf-8"))
        validation_history = json.loads(
            validation_history_path.read_text(encoding="utf-8")
        )
        if not history or int(history[-1]["episode"]) != completed_episode:
            raise ValueError("training history does not end at the checkpoint episode")

        # Recreate the deterministic shuffled instance order for the cycle that
        # contains the first resumed episode. The checkpoint is written only at
        # rollout-update boundaries, so no partial rollout needs restoration.
        completed_cycles = (completed_episode - 1) // len(order) + 1
        for _ in range(completed_cycles):
            sampler.shuffle(order)
        print(
            f"resuming from episode={completed_episode:04d} "
            f"total_environment_steps={total_environment_steps}"
        )

    for episode in range(start_episode, episodes + 1):
        position = (episode - 1) % len(order)
        if position == 0:
            sampler.shuffle(order)
        train_instance = train_instances[order[position]]
        env = ResidualStressFJSPEnv(
            train_instance,
            observation_profile=observation_profile,
            action_feature_profile=action_feature_profile,
        )
        episode_buffer, metrics = collect_episode(env, model, seed=seed + episode)
        rollout.extend(episode_buffer)
        total_environment_steps += len(episode_buffer)
        row = {
            "episode": float(episode),
            "instance": train_instance.name,
            "objective": float(metrics["objective"]),
            "reward": float(sum(episode_buffer.rewards)),
            "episode_steps": len(episode_buffer),
            "total_environment_steps": total_environment_steps,
            "policy_loss": None,
            "value_loss": None,
            "entropy": None,
            "learning_rate": algorithm.learning_rate,
            "entropy_coef": algorithm.entropy_coef,
        }
        history.append(row)
        update_due = episode % episodes_per_update == 0 or episode == episodes
        if not update_due:
            continue

        schedule_progress = (
            (episode - episodes_per_update)
            / max(1, episodes - episodes_per_update)
        )
        schedule_progress = min(1.0, max(0.0, schedule_progress))
        algorithm.set_learning_rate(
            initial_learning_rate
            + schedule_progress * (final_learning_rate - initial_learning_rate)
        )
        algorithm.set_entropy_coef(
            initial_entropy_coef
            + schedule_progress * (final_entropy_coef - initial_entropy_coef)
        )
        losses = algorithm.update(rollout)
        row.update(losses)
        row["learning_rate"] = algorithm.learning_rate
        row["entropy_coef"] = algorithm.entropy_coef
        row["rollout_steps"] = len(rollout)
        rollout = RolloutBuffer()

        validation_due = episode % validation_interval == 0 or episode == episodes
        if validation_due:
            latest_evaluation, latest_schedules = _evaluate_validation_set(
                eval_instances,
                model,
                seed=int(config.get("eval_seed", seed + 1000)),
                observation_profile=observation_profile,
                action_feature_profile=action_feature_profile,
            )
            validation_history.append(
                {
                    "episode": episode,
                    "total_environment_steps": total_environment_steps,
                    "objective": float(latest_evaluation["objective"]),
                    "terminated": bool(latest_evaluation["terminated"]),
                    "truncated": bool(latest_evaluation["truncated"]),
                }
            )
            if (
                _is_valid_evaluation(latest_evaluation)
                and float(latest_evaluation["objective"]) < best_validation_objective
            ):
                best_validation_objective = float(latest_evaluation["objective"])
                _save_checkpoint(
                    checkpoints / "best.pt",
                    episode=episode,
                    total_environment_steps=total_environment_steps,
                    best_validation_objective=best_validation_objective,
                    **common_checkpoint_args,
                )
                write_json(output / "best_evaluation.json", latest_evaluation)

        checkpoint_args = {
            "episode": episode,
            "total_environment_steps": total_environment_steps,
            "best_validation_objective": best_validation_objective,
            **common_checkpoint_args,
        }
        if episode % latest_interval == 0 or episode == episodes:
            _save_checkpoint(checkpoints / "latest.pt", **checkpoint_args)
        if episode % periodic_interval == 0:
            _save_checkpoint(checkpoints / f"episode_{episode:04d}.pt", **checkpoint_args)

        write_json(output / "training_history.json", history)
        write_json(output / "validation_history.json", validation_history)
        if episode == episodes_per_update or episode % validation_interval == 0 or episode == episodes:
            validation_text = (
                f" validation={latest_evaluation['objective']:.4f}"
                if validation_due and latest_evaluation is not None
                else ""
            )
            print(
                f"episode={episode:04d} instance={train_instance.name} "
                f"objective={row['objective']:.4f} "
                f"rollout_steps={row['rollout_steps']} "
                f"policy_loss={row['policy_loss']:.4f}{validation_text}"
            )

    if latest_evaluation is None or latest_schedules is None:
        latest_evaluation, latest_schedules = _evaluate_validation_set(
            eval_instances,
            model,
            seed=int(config.get("eval_seed", seed + 1000)),
            observation_profile=observation_profile,
            action_feature_profile=action_feature_profile,
        )
    evaluation = latest_evaluation
    validation_schedules = latest_schedules
    first_schedule = validation_schedules[eval_instances[0].name]
    final_checkpoint_args = {
        "episode": episodes,
        "total_environment_steps": total_environment_steps,
        "best_validation_objective": best_validation_objective,
        **common_checkpoint_args,
    }
    _save_checkpoint(checkpoints / "final.pt", **final_checkpoint_args)
    _save_checkpoint(output / "model.pt", **final_checkpoint_args)
    write_json(
        output / "checkpoint_summary.json",
        {
            "latest": "checkpoints/latest.pt",
            "best": "checkpoints/best.pt",
            "final": "checkpoints/final.pt",
            "best_validation_objective": best_validation_objective,
            "total_environment_steps": total_environment_steps,
        },
    )
    write_json(output / "config.json", config)
    write_json(output / "training_history.json", history)
    write_json(output / "validation_history.json", validation_history)
    write_json(output / "evaluation.json", evaluation)
    write_json(output / "validation_schedules.json", validation_schedules)
    write_json(output / "schedule.json", first_schedule)
    if bool(config.get("enable_plots", True)):
        try:
            plot_training_curve(history, output / "training_curve.png")
            plot_gantt(first_schedule, output / "gantt.png")
        except (RuntimeError, OSError) as exc:
            print(f"plot skipped: {exc}")
    return model, history, evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--resume",
        default=None,
        help="checkpoint to resume from; existing histories must match its episode",
    )
    parser.add_argument(
        "--initialize-from",
        default=None,
        help="load model weights only and start a new run with a fresh optimizer",
    )
    args = parser.parse_args()
    config = load_run_config(args.config)
    initialize_from = args.initialize_from or config.get("initialize_from")
    _, _, evaluation = train(
        config,
        resume_from=args.resume,
        initialize_from=initialize_from,
    )
    print(f"evaluation objective={evaluation['objective']:.4f}")


if __name__ == "__main__":
    main()

