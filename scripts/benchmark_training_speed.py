"""Measure a short, fixed PPO workload on CPU or with torchrun.

This is a throughput diagnostic, not a convergence experiment. Input settings
for the model, rollout size, global mini-batch and PPO epochs are preserved;
learning rate and entropy stay at their configured initial values. Validation
runs once at the end, with a fixed prefix of the configured validation files.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import platform
import sys
import time
from uuid import uuid4

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppo_rs.cli_utils import choose_device, load_run_config, write_json  # noqa: E402
from ppo_rs.config import resolve_instance_paths  # noqa: E402
from ppo_rs.train import train  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True, help="parent for a new unique benchmark directory")
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--validation-instances", type=int, default=16)
    parser.add_argument("--device", help="override config device, for example cpu or cuda")
    parser.add_argument("--checkpoint", help="initialize model weights without resuming an existing run")
    args = parser.parse_args()
    if args.updates <= 0 or args.validation_instances <= 0:
        parser.error("updates and validation-instances must be positive")

    config = load_run_config(args.config)
    if args.device:
        config["device"] = args.device
    device = choose_device(str(config.get("device", "auto")))
    owns_process_group = False
    try:
        if int(os.environ.get("WORLD_SIZE", "1")) > 1 and not dist.is_initialized():
            if device.type == "cuda":
                torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
            dist.init_process_group(
                backend="nccl" if device.type == "cuda" else "gloo",
                timeout=timedelta(minutes=30),
            )
            owns_process_group = True
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        if device.type == "cuda" and world_size > 1:
            device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        destination = None
        if rank == 0:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            destination = str(Path(args.output).expanduser().resolve() / f"benchmark_{timestamp}_{uuid4().hex[:8]}")
        if world_size > 1:
            shared = [destination]
            dist.broadcast_object_list(shared, src=0)
            destination = shared[0]
        assert destination is not None

        source = config.get("eval_instances", config.get("eval_instance"))
        if source is None:
            source = config.get("train_instances", config.get("train_instance"))
        paths = resolve_instance_paths(source)
        selected_paths = [str(path.resolve()) for path in paths[: args.validation_instances]]
        rollout_episodes = int(config.get("episodes_per_update", 1))
        if rollout_episodes <= 0:
            raise ValueError("episodes_per_update must be positive")
        episodes = args.updates * rollout_episodes
        config.update(
            episodes=episodes,
            validation_interval=episodes,
            latest_checkpoint_interval=episodes,
            checkpoint_interval=episodes,
            output_dir=destination,
            output_dir_mode="unique",
            distributed_world_size=world_size,
            eval_instances=selected_paths,
            enable_plots=False,
            final_learning_rate=float(config.get("learning_rate", 3e-4)),
            final_entropy_coef=float(config.get("entropy_coef", 0.01)),
        )
        # A short benchmark never resumes or silently inherits a fine-tune
        # source. The optional argument explicitly controls model initialization.
        config.pop("initialize_from", None)
        if world_size > 1:
            dist.barrier()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        _, history, evaluation = train(config, initialize_from=args.checkpoint)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        wall_seconds = time.perf_counter() - started
        if world_size > 1:
            duration = torch.tensor(wall_seconds, dtype=torch.float64, device=device if dist.get_backend() == "nccl" else "cpu")
            dist.all_reduce(duration, op=dist.ReduceOp.MAX)
            wall_seconds = float(duration.item())
        if rank == 0:
            runs = [path for path in Path(destination).iterdir() if path.is_dir()]
            if len(runs) != 1:
                raise RuntimeError("benchmark expected exactly one unique training run")
            run = runs[0]
            steps = int(history[-1]["total_environment_steps"])
            gpu_count = world_size if device.type == "cuda" else 0
            report = {
                "purpose": "throughput diagnostic; not a convergence or model-quality comparison",
                "source_config": str(Path(args.config).resolve()),
                "run_directory": str(run),
                "checkpoint": str(Path(args.checkpoint).resolve()) if args.checkpoint else None,
                "world_size": world_size,
                "device": str(device),
                "updates": args.updates,
                "episodes": len(history),
                "environment_steps": steps,
                "wall_seconds": wall_seconds,
                "environment_steps_per_second": steps / wall_seconds,
                "episodes_per_second": len(history) / wall_seconds,
                "ppo_update_wall_seconds": sum(float(row.get("ppo_update_wall_seconds", 0.0)) for row in history),
                "validation_wall_seconds": float(evaluation["validation_wall_seconds"]),
                "validation_instance_count": len(selected_paths),
                "validation_objective": float(evaluation["objective"]),
                "validation_terminated": bool(evaluation["terminated"]),
                "validation_truncated": bool(evaluation["truncated"]),
                "allocated_gpu_hours": gpu_count * wall_seconds / 3600.0,
                "hardware": {
                    "platform": platform.platform(),
                    "processor": platform.processor(),
                    "cpu_count": os.cpu_count(),
                    "torch_version": torch.__version__,
                    "cuda_runtime": torch.version.cuda,
                    "torch_num_threads": torch.get_num_threads(),
                    "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
                    "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
                    "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                },
                "notes": [
                    "Wall time includes input/model initialization, all updates, final validation and normal training file output.",
                    "No separate warm-up is performed; repeat identical runs to assess cold-start effects.",
                    "Learning rate and entropy are fixed at the source config's initial values for this short diagnostic.",
                    "The original config, checkpoints and output directory are not modified.",
                ],
            }
            write_json(run / "benchmark.json", report)
            print(json.dumps(report, indent=2))
    finally:
        if owns_process_group:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
