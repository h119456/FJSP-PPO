"""Evaluate a trained PPO checkpoint and all rule baselines."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ppo_rs import load_instance
from ppo_rs.baselines import BASELINES, run_baseline
from ppo_rs.cli_utils import choose_device, write_json
from ppo_rs.environment import ResidualStressFJSPEnv
from ppo_rs.evaluation import evaluate_model, plot_gantt
from ppo_rs.models import build_model_from_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", default="data/examples/synthetic_test.json")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default="outputs/evaluation")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1007)
    args = parser.parse_args()
    instance = load_instance(args.instance)
    output = Path(args.output_dir)
    results: list[dict] = []
    for name in BASELINES:
        metrics, schedule = run_baseline(ResidualStressFJSPEnv(instance), name, seed=args.seed)
        results.append(metrics)
        write_json(output / f"{name}_schedule.json", schedule)
        try:
            plot_gantt(schedule, output / f"{name}_gantt.png")
        except RuntimeError as exc:
            print(f"plot skipped: {exc}")

    if args.checkpoint:
        device = choose_device(args.device)
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
        checkpoint_version = int(checkpoint.get("environment_semantics_version", 1))
        current_version = ResidualStressFJSPEnv.ENVIRONMENT_SEMANTICS_VERSION
        if checkpoint_version != current_version:
            raise RuntimeError(
                "checkpoint environment semantics are incompatible: "
                f"checkpoint={checkpoint_version}, current={current_version}; retrain the model"
            )
        model = build_model_from_checkpoint(checkpoint, device)
        observation_profile = str(
            checkpoint.get(
                "observation_profile",
                checkpoint.get("config", {}).get(
                    "observation_profile", "compact14"
                ),
            )
        )
        action_feature_profile = str(
            checkpoint.get(
                "action_feature_profile",
                checkpoint.get("config", {}).get(
                    "action_feature_profile", "compact21"
                ),
            )
        )
        metrics, schedule = evaluate_model(
            ResidualStressFJSPEnv(
                instance,
                observation_profile=observation_profile,
                action_feature_profile=action_feature_profile,
            ),
            model,
            seed=args.seed,
        )
        results.append(metrics)
        write_json(output / "ppo_schedule.json", schedule)
        try:
            plot_gantt(schedule, output / "ppo_gantt.png")
        except RuntimeError as exc:
            print(f"plot skipped: {exc}")
    write_json(output / "metrics.json", results)
    for row in results:
        print(f"{row['policy']:>22s} objective={row['objective']:.4f} makespan={row['makespan']:.4f}")


if __name__ == "__main__":
    main()
