"""Compare trained checkpoints across the same instance splits."""

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
from ppo_rs.environment import ResidualStressFJSPEnv  # noqa: E402
from ppo_rs.environment.checkpoint_semantics import check_dynamic_input_semantics  # noqa: E402
from ppo_rs.evaluation import evaluate_model  # noqa: E402
from ppo_rs.models import build_model_from_checkpoint  # noqa: E402


MEAN_METRICS = (
    "objective",
    "makespan",
    "normal_processing_time",
    "inter_operation_idle_time",
    "stress_relief_count",
    "compensation_count",
    "total_independent_cost",
)


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    result = {
        "instance_count": count,
        **{
            key: sum(float(row[key]) for row in rows) / count
            for key in MEAN_METRICS
        },
        "schedule_violation_count": sum(
            int(row["schedule_violation_count"]) for row in rows
        ),
        "process_readiness_violations": sum(
            int(row["process_readiness_violations"]) for row in rows
        ),
        "truncated_count": sum(bool(row["truncated"]) for row in rows),
    }
    for key in (
        "job_release_event_count",
        "machine_breakdown_event_count",
        "interrupted_operation_count",
        "total_repair_time",
        "total_interruption_time",
        "episode_wall_seconds",
        "policy_inference_seconds",
        "graph_construction_seconds",
    ):
        if all(key in row for row in rows):
            result[key] = sum(float(row[key]) for row in rows) / count
    if all("episode_wall_seconds" in row for row in rows):
        result["total_episode_wall_seconds"] = sum(float(row["episode_wall_seconds"]) for row in rows)
    return result


def _mapping(values: list[str], option: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{option} needs LABEL=PATH, got {value!r}")
        label, path = value.split("=", 1)
        if not label or not path:
            raise ValueError(f"{option} needs non-empty LABEL=PATH")
        result[label] = path
    return result


def _evaluate_split(
    source: str,
    model: torch.nn.Module,
    observation_profile: str,
    action_feature_profile: str,
    seed: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    scenario_rows: dict[str, list[dict[str, Any]]] = {}
    for index, instance in enumerate(load_instances(source)):
        environment = ResidualStressFJSPEnv(
            instance,
            observation_profile=observation_profile,
            action_feature_profile=action_feature_profile,
        )
        metrics, _ = evaluate_model(
            environment, model, seed=seed + index, deterministic=True
        )
        rows.append(metrics)
        scenario = instance.dynamic_events.get("scenario")
        if scenario is not None:
            scenario_rows.setdefault(str(scenario), []).append(metrics)
    result = _summarize_rows(rows)
    if scenario_rows:
        result["scenarios"] = {
            scenario: _summarize_rows(values)
            for scenario, values in sorted(scenario_rows.items())
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", action="append", required=True, help="LABEL=CHECKPOINT"
    )
    parser.add_argument(
        "--split", action="append", required=True, help="LABEL=INSTANCE_DIRECTORY"
    )
    parser.add_argument("--seed", type=int, default=1007)
    parser.add_argument("--output")
    args = parser.parse_args()

    runs = _mapping(args.run, "--run")
    splits = _mapping(args.split, "--split")
    summary: dict[str, Any] = {}
    for run_label, checkpoint_path in runs.items():
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        check_dynamic_input_semantics(
            checkpoint, dynamic=any(i.dynamic_events for source in splits.values() for i in load_instances(source))
        )
        model = build_model_from_checkpoint(checkpoint, torch.device("cpu"))
        profile = str(
            checkpoint.get(
                "observation_profile",
                checkpoint.get("config", {}).get(
                    "observation_profile", "compact14"
                ),
            )
        )
        action_profile = str(
            checkpoint.get(
                "action_feature_profile",
                checkpoint.get("config", {}).get(
                    "action_feature_profile", "compact21"
                ),
            )
        )
        summary[run_label] = {
            "checkpoint": checkpoint_path,
            "dynamic_input_semantics_version": checkpoint.get("dynamic_input_semantics_version"),
            "evaluation_dynamic_input_semantics_version": ResidualStressFJSPEnv.DYNAMIC_INPUT_SEMANTICS_VERSION,
            "episode": int(checkpoint["episode"]),
            "observation_profile": profile,
            "observation_dim": int(checkpoint["observation_dim"]),
            "action_feature_profile": action_profile,
            "action_feature_dim": int(checkpoint["action_feature_dim"]),
            "splits": {
                split_label: _evaluate_split(
                    source, model, profile, action_profile, args.seed
                )
                for split_label, source in splits.items()
            },
        }

    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
