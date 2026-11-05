"""Reproducible PPO training entry point."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo

import torch
import torch.distributed as dist
from torch import nn

from ppo_rs import load_instances
from ppo_rs.algorithms import PPO, PPOConfig, RolloutBuffer, collect_episode
from ppo_rs.cli_utils import choose_device, load_run_config, set_seed, write_json
from ppo_rs.environment import ResidualStressFJSPEnv
from ppo_rs.environment.checkpoint_semantics import check_dynamic_input_semantics
from ppo_rs.evaluation import evaluate_model, plot_gantt, plot_training_curve
from ppo_rs.models import build_actor_critic


def _distributed_context() -> tuple[int, int]:
    return (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)


def _broadcast_object(value: Any) -> Any:
    if dist.is_initialized() and dist.get_world_size() > 1:
        values = [value]
        dist.broadcast_object_list(values, src=0)
        return values[0]
    return value


@contextmanager
def _rank_zero_phase():
    """Propagate root-only validation/I/O failures to every waiting rank."""
    error = None
    try:
        yield
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    error = _broadcast_object(error)
    if error is not None:
        raise RuntimeError(f"rank0 operation failed: {error}")


def _validate_transfer_schema(checkpoint: dict[str, Any], model: nn.Module,
                              observation_profile: str, action_feature_profile: str) -> None:
    if int(checkpoint.get("environment_semantics_version", 1)) != ResidualStressFJSPEnv.ENVIRONMENT_SEMANTICS_VERSION:
        raise ValueError("checkpoint environment semantics do not match the current environment")
    previous = checkpoint.get("config", {})
    if checkpoint.get("observation_profile", previous.get("observation_profile", "compact14")) != observation_profile:
        raise ValueError("checkpoint observation profile does not match the current environment")
    if checkpoint.get("action_feature_profile", previous.get("action_feature_profile", "compact21")) != action_feature_profile:
        raise ValueError("checkpoint action feature profile does not match the current environment")
    if dict(checkpoint.get("model_kwargs", {})) != dict(getattr(model, "checkpoint_kwargs", {})):
        raise ValueError("checkpoint graph architecture does not match the current model")


def _run_directory(config: dict[str, Any], resume_from: str | Path | None) -> Path:
    """Allocate a shared, collision-free run directory on rank zero."""
    rank, _ = _distributed_context()
    output = None
    with _rank_zero_phase():
        if rank == 0:
            if resume_from is not None:
                checkpoint = Path(resume_from).resolve()
                output = checkpoint.parent.parent if checkpoint.parent.name == "checkpoints" else checkpoint.parent
            else:
                root = Path(config.get("output_dir", "outputs/default_run")).expanduser().resolve()
                mode = str(config.get("output_dir_mode", "existing"))
                if mode not in {"existing", "unique"}:
                    raise ValueError("output_dir_mode must be 'existing' or 'unique'")
                if mode == "unique":
                    timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
                    name = f"{timestamp}_{config.get('model_type', 'mlp')}_seed{int(config.get('seed', 7))}_{uuid4().hex[:8]}"
                    root.mkdir(parents=True, exist_ok=True)
                    output = root / name
                    output.mkdir(exist_ok=False)
                else:
                    output = root
                    output.mkdir(parents=True, exist_ok=True)
    return Path(_broadcast_object(str(output) if output is not None else None))


def _episode_order(instance_count: int, episodes: int, seed: int) -> list[int]:
    order = list(range(instance_count))
    sampler = random.Random(seed)
    result = []
    for index in range(episodes):
        if index % instance_count == 0:
            sampler.shuffle(order)
        result.append(order[index % instance_count])
    return result


def _random_states(device: torch.device) -> list[dict[str, Any]]:
    state = {"torch": torch.get_rng_state()}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device).cpu()
    _, world_size = _distributed_context()
    if world_size == 1:
        return [state]
    states: list[Any] = [None] * world_size
    dist.all_gather_object(states, state)
    return states


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
    """Evaluate disjoint instance shards, then restore their original order.

    All ranks participate, including ranks with no assigned instances. Policy
    evaluation is deterministic and environment resets use their own RNG, so
    distributing validation does not consume the training policy's RNG stream.
    """

    validation_started = time.perf_counter()
    rank, world_size = _distributed_context()
    local_results: list[tuple[int, dict[str, Any], list[dict[str, Any]]]] = []
    error = None
    try:
        if not instances:
            raise ValueError("validation requires at least one instance")
        for index in range(rank, len(instances), world_size):
            instance = instances[index]
            metrics, schedule = evaluate_model(
                ResidualStressFJSPEnv(
                    instance,
                    observation_profile=observation_profile,
                    action_feature_profile=action_feature_profile,
                ),
                model,
                seed=seed + index,
            )
            local_results.append((index, {"name": instance.name, "metrics": metrics}, schedule))
    except Exception as exc:
        if world_size == 1:
            raise
        error = f"{type(exc).__name__}: {exc}"

    if world_size > 1:
        # Propagate failures in the same collective as the results. Raising on
        # only one rank before this point would leave its peers waiting forever.
        gathered: list[Any] = [None] * world_size
        dist.all_gather_object(gathered, {"error": error, "results": local_results})
        errors = [f"rank{index}: {item['error']}" for index, item in enumerate(gathered) if item["error"]]
        if errors:
            raise RuntimeError("validation failed: " + "; ".join(errors))
        ordered = sorted((row for item in gathered for row in item["results"]), key=lambda row: row[0])
    else:
        ordered = local_results
    if [row[0] for row in ordered] != list(range(len(instances))):
        raise RuntimeError("validation did not evaluate every instance exactly once")
    results = [row[1] for row in ordered]
    schedules = {row[1]["name"]: row[2] for row in ordered}
    aggregate = _mean_validation_metrics(results)
    # The parallel pass includes construction, schedule extraction, result
    # gathering and aggregation. Its wall time is the slowest rank, not the sum
    # of per-instance episode timings, which may overlap across devices.
    validation_seconds = time.perf_counter() - validation_started
    if world_size > 1:
        timing_device = getattr(model, "device", "cpu") if dist.get_backend() == "nccl" else "cpu"
        duration = torch.tensor(validation_seconds, dtype=torch.float64, device=timing_device)
        dist.all_reduce(duration, op=dist.ReduceOp.MAX)
        validation_seconds = float(duration.item())
    aggregate["validation_wall_seconds"] = validation_seconds
    return aggregate, schedules


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
    distributed_random_states: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Create an evaluation-compatible checkpoint that can also resume training."""

    return {
        "environment_semantics_version": ResidualStressFJSPEnv.ENVIRONMENT_SEMANTICS_VERSION,
        "dynamic_input_semantics_version": ResidualStressFJSPEnv.DYNAMIC_INPUT_SEMANTICS_VERSION,
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
        "distributed_random_states": distributed_random_states,
        "distributed_world_size": _distributed_context()[1],
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
    config = dict(config)
    rank, world_size = _distributed_context()
    expected_world_size = int(config.get("distributed_world_size", world_size))
    if expected_world_size != world_size:
        raise ValueError(
            f"config requires {expected_world_size} processes; received {world_size}. "
            f"Launch with torchrun --standalone --nproc_per_node={expected_world_size} -m ppo_rs.train ..."
        )
    seed = int(config.get("seed", 7))
    device = choose_device(str(config.get("device", "auto")))
    if world_size > 1 and device.type == "cuda":
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", rank)))
        torch.cuda.set_device(device)
    set_seed(seed)
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
    dynamic_inputs = any(instance.dynamic_events for instance in train_instances + eval_instances)
    output = _run_directory(config, resume_from)
    config["output_dir"] = str(output)
    if initialize_from is not None:
        config["initialize_from"] = str(Path(initialize_from).resolve())
    config["distributed_world_size"] = world_size
    config["global_mini_batch_size"] = ppo_config.mini_batch_size
    config["per_rank_mini_batch_size"] = (ppo_config.mini_batch_size + world_size - 1) // world_size
    checkpoints = output / "checkpoints"
    with _rank_zero_phase():
        if rank == 0:
            write_json(output / "config.json", config)
            print(f"run_directory={output} world_size={world_size} global_mini_batch_size={ppo_config.mini_batch_size}", flush=True)

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
    episode_order = _episode_order(len(train_instances), episodes, seed)
    if world_size > 1:
        set_seed(seed + rank * 1000003)
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
        check_dynamic_input_semantics(checkpoint, dynamic=dynamic_inputs)
        _validate_transfer_schema(checkpoint, model, observation_profile, action_feature_profile)
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
        if rank == 0:
            print(
                f"initialized model weights from {initialization_path} "
                f"(source_episode={int(checkpoint['episode'])}); optimizer reset"
            )

    if resume_from is not None:
        resume_path = Path(resume_from)
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        check_dynamic_input_semantics(checkpoint, dynamic=dynamic_inputs, resume=True)
        if int(checkpoint["environment_semantics_version"]) != int(
            ResidualStressFJSPEnv.ENVIRONMENT_SEMANTICS_VERSION
        ):
            raise ValueError("checkpoint environment semantics do not match the current environment")
        _validate_transfer_schema(checkpoint, model, observation_profile, action_feature_profile)
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
        saved_states = checkpoint.get("distributed_random_states")
        if saved_states is not None:
            if len(saved_states) != world_size:
                raise ValueError("resume requires the checkpoint's original distributed world size")
            torch.set_rng_state(saved_states[rank]["torch"].cpu())
            if device.type == "cuda" and "cuda" in saved_states[rank]:
                torch.cuda.set_rng_state(saved_states[rank]["cuda"].cpu(), device)
        else:
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

        if rank == 0:
            print(
                f"resuming from episode={completed_episode:04d} "
                f"total_environment_steps={total_environment_steps}"
            )

    for block_start in range(start_episode, episodes + 1, episodes_per_update):
        episode = min(episodes, block_start + episodes_per_update - 1)
        local_episodes = []
        for current_episode in range(block_start, episode + 1):
            if (current_episode - 1) % world_size != rank:
                continue
            train_instance = train_instances[episode_order[current_episode - 1]]
            env = ResidualStressFJSPEnv(
                train_instance,
                observation_profile=observation_profile,
                action_feature_profile=action_feature_profile,
            )
            episode_buffer, metrics = collect_episode(env, model, seed=seed + current_episode)
            local_episodes.append((current_episode, train_instance.name, episode_buffer, metrics))
        if world_size > 1:
            gathered: list[Any] = [None] * world_size
            dist.all_gather_object(gathered, local_episodes)
            collected = sorted((item for values in gathered for item in values), key=lambda item: item[0])
        else:
            collected = local_episodes
        rollout = RolloutBuffer()
        for current_episode, instance_name, episode_buffer, metrics in collected:
            rollout.extend(episode_buffer)
            total_environment_steps += len(episode_buffer)
            row = {
                "episode": float(current_episode),
                "instance": instance_name,
                "objective": float(metrics["objective"]),
                "reward": float(sum(episode_buffer.rewards)),
                "episode_steps": len(episode_buffer),
                "terminated": bool(metrics["terminated"]),
                "truncated": bool(metrics["truncated"]),
                "episode_wall_seconds": float(metrics["episode_wall_seconds"]),
                "policy_inference_seconds": float(metrics["policy_inference_seconds"]),
                "graph_construction_seconds": float(metrics["graph_construction_seconds"]),
                "total_environment_steps": total_environment_steps,
                "policy_loss": None,
                "value_loss": None,
                "entropy": None,
                "learning_rate": algorithm.learning_rate,
                "entropy_coef": algorithm.entropy_coef,
                "collection_rank": (current_episode - 1) % world_size,
            }
            history.append(row)
        if len(collected) != episode - block_start + 1:
            raise RuntimeError("distributed rollout did not collect every global episode exactly once")

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
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        update_started = time.perf_counter()
        losses = algorithm.update(rollout)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        update_seconds = time.perf_counter() - update_started
        if world_size > 1:
            # Wall time of the shared update is determined by the slowest rank;
            # summing ranks would misrepresent parallel work as elapsed time.
            duration = torch.tensor(update_seconds, dtype=torch.float64, device=device)
            dist.all_reduce(duration, op=dist.ReduceOp.MAX)
            update_seconds = float(duration.item())
        row.update(losses)
        # One optimizer update belongs to a whole rollout block, not each job
        # instance. Record it only on the final episode row of that block.
        row["ppo_update_wall_seconds"] = update_seconds
        row["learning_rate"] = algorithm.learning_rate
        row["entropy_coef"] = algorithm.entropy_coef
        row["rollout_steps"] = len(rollout)
        common_checkpoint_args["distributed_random_states"] = _random_states(device)

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
                    "episode_wall_seconds": float(latest_evaluation["episode_wall_seconds"]),
                    "policy_inference_seconds": float(latest_evaluation["policy_inference_seconds"]),
                    "graph_construction_seconds": float(latest_evaluation["graph_construction_seconds"]),
                    "validation_wall_seconds": float(latest_evaluation["validation_wall_seconds"]),
                }
            )
            if (
                _is_valid_evaluation(latest_evaluation)
                and float(latest_evaluation["objective"]) < best_validation_objective
            ):
                best_validation_objective = float(latest_evaluation["objective"])
                with _rank_zero_phase():
                    if rank == 0:
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
        with _rank_zero_phase():
            if rank == 0:
                if episode % latest_interval == 0 or episode == episodes:
                    _save_checkpoint(checkpoints / "latest.pt", **checkpoint_args)
                if episode % periodic_interval == 0:
                    _save_checkpoint(checkpoints / f"episode_{episode:04d}.pt", **checkpoint_args)

                write_json(output / "training_history.json", history)
                write_json(output / "validation_history.json", validation_history)
                if episode == episodes_per_update or episode % validation_interval == 0 or episode == episodes:
                    validation_text = (
                        f" validation={latest_evaluation['objective']:.4f}"
                        f" validation_seconds={latest_evaluation['validation_wall_seconds']:.3f}"
                        if validation_due and latest_evaluation is not None
                        else ""
                    )
                    print(
                        f"episode={episode:04d} instance={row['instance']} "
                        f"objective={row['objective']:.4f} "
                        f"rollout_steps={row['rollout_steps']} "
                        f"policy_loss={row['policy_loss']:.4f} "
                        f"episode_seconds={row['episode_wall_seconds']:.3f} "
                        f"ppo_update_seconds={row['ppo_update_wall_seconds']:.3f}{validation_text}"
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
    with _rank_zero_phase():
        if rank == 0:
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
    if world_size > 1:
        dist.barrier()
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
    config.setdefault("output_dir_mode", "unique")
    initialize_from = args.initialize_from or config.get("initialize_from")
    owns_process_group = False
    try:
        if int(os.environ.get("WORLD_SIZE", "1")) > 1 and not dist.is_initialized():
            use_cuda = str(config.get("device", "auto")) != "cpu" and torch.cuda.is_available()
            if use_cuda:
                torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
            dist.init_process_group(
                backend="nccl" if use_cuda else "gloo",
                timeout=timedelta(minutes=30),
            )
            owns_process_group = True
        _, _, evaluation = train(
            config,
            resume_from=args.resume,
            initialize_from=initialize_from,
        )
        if _distributed_context()[0] == 0:
            print(f"evaluation objective={evaluation['objective']:.4f}")
    finally:
        if owns_process_group:
            dist.destroy_process_group()



if __name__ == "__main__":
    main()

