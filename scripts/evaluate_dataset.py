"""Evaluate PPO checkpoints and rule baselines over a complete dataset split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppo_rs import load_instances  # noqa: E402
from ppo_rs.baselines import BASELINES, run_baseline  # noqa: E402
from ppo_rs.cli_utils import choose_device, write_json  # noqa: E402
from ppo_rs.environment import ResidualStressFJSPEnv  # noqa: E402
from ppo_rs.evaluation import evaluate_model  # noqa: E402
from ppo_rs.models import build_model_from_checkpoint  # noqa: E402


def _load_model(path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    version = int(checkpoint.get("environment_semantics_version", 1))
    if version != ResidualStressFJSPEnv.ENVIRONMENT_SEMANTICS_VERSION:
        raise RuntimeError(
            f"incompatible checkpoint semantics: checkpoint={version}, "
            f"current={ResidualStressFJSPEnv.ENVIRONMENT_SEMANTICS_VERSION}"
        )
    model = build_model_from_checkpoint(checkpoint, device)
    return model, checkpoint


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_keys = {
        key
        for key in rows[0]
        if all(
            isinstance(row.get(key), (int, float))
            and not isinstance(row.get(key), bool)
            for row in rows
        )
    }
    result = {
        f"mean_{key}": sum(float(row[key]) for row in rows) / len(rows)
        for key in sorted(numeric_keys)
    }
    result.update(
        {
            "instance_count": len(rows),
            "valid_instance_count": sum(
                bool(row["terminated"])
                and not bool(row["truncated"])
                and int(row["schedule_violation_count"]) == 0
                for row in rows
            ),
            "total_schedule_violations": sum(
                int(row["schedule_violation_count"]) for row in rows
            ),
            "total_process_readiness_violations": sum(
                int(row["process_readiness_violations"]) for row in rows
            ),
        }
    )
    return result


def _checkpoint_argument(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("checkpoint must use NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("checkpoint name cannot be empty")
    return name, Path(raw_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", required=True)
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=_checkpoint_argument,
        default=[],
        metavar="NAME=PATH",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1007)
    args = parser.parse_args()

    device = choose_device(args.device)
    instances = load_instances(args.instances)
    models: dict[str, torch.nn.Module] = {}
    checkpoint_metadata: dict[str, Any] = {}
    for name, path in args.checkpoint:
        model, checkpoint = _load_model(path, device)
        models[name] = model
        checkpoint_metadata[name] = {
            "path": str(path),
            "episode": int(checkpoint["episode"]),
            "total_environment_steps": int(checkpoint["total_environment_steps"]),
        }

    policy_rows: dict[str, list[dict[str, Any]]] = {
        **{name: [] for name in BASELINES},
        **{name: [] for name in models},
    }
    per_instance: list[dict[str, Any]] = []
    for index, instance in enumerate(instances):
        instance_result: dict[str, Any] = {"name": instance.name, "policies": {}}
        for name in BASELINES:
            metrics, _ = run_baseline(
                ResidualStressFJSPEnv(instance), name, seed=args.seed + index
            )
            policy_rows[name].append(metrics)
            instance_result["policies"][name] = metrics
        for name, model in models.items():
            metrics, _ = evaluate_model(
                ResidualStressFJSPEnv(instance), model, seed=args.seed + index
            )
            metrics["policy"] = name
            policy_rows[name].append(metrics)
            instance_result["policies"][name] = metrics
        per_instance.append(instance_result)
        if (index + 1) % 25 == 0 or index + 1 == len(instances):
            print(f"evaluated {index + 1}/{len(instances)} instances", flush=True)

    aggregates = {name: _aggregate(rows) for name, rows in policy_rows.items()}
    best_baseline = min(BASELINES, key=lambda name: aggregates[name]["mean_objective"])
    comparisons: dict[str, Any] = {"best_rule_baseline": best_baseline}
    baseline_objectives = [
        row["objective"] for row in policy_rows[best_baseline]
    ]
    baseline_mean = float(aggregates[best_baseline]["mean_objective"])
    for name in models:
        model_objectives = [row["objective"] for row in policy_rows[name]]
        model_mean = float(aggregates[name]["mean_objective"])
        comparisons[name] = {
            "objective_improvement_percent_vs_best_rule": (
                100.0 * (baseline_mean - model_mean) / baseline_mean
            ),
            "paired_win_count_vs_best_rule": sum(
                model_value < baseline_value - 1e-9
                for model_value, baseline_value in zip(
                    model_objectives, baseline_objectives
                )
            ),
            "paired_tie_count_vs_best_rule": sum(
                abs(model_value - baseline_value) <= 1e-9
                for model_value, baseline_value in zip(
                    model_objectives, baseline_objectives
                )
            ),
            "paired_loss_count_vs_best_rule": sum(
                model_value > baseline_value + 1e-9
                for model_value, baseline_value in zip(
                    model_objectives, baseline_objectives
                )
            ),
        }

    result = {
        "instance_source": args.instances,
        "seed": args.seed,
        "checkpoint_metadata": checkpoint_metadata,
        "aggregates": aggregates,
        "comparisons": comparisons,
        "instances": per_instance,
    }
    write_json(args.output, result)
    print(json.dumps({"aggregates": aggregates, "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
