"""Check or launch matched base/full state experiments without replacing runs.

Each selected model uses all requested processes; ``both`` runs MLP then GNN.
Dynamic fine-tuning loads the corresponding static weights and starts a new
optimizer, schedule and history. Use ``--check`` for a CPU-only preflight.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any
from uuid import uuid4

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from ppo_rs.cli_utils import load_run_config, write_json  # noqa: E402
from scripts.check_comparison import (  # noqa: E402
    _constraints,
    _dataset_summary,
    _model_summary,
    audit_comparison,
    effective_config,
)


DATASETS = (
    "dataset_test", "dataset1", "dataset1_cost0", "dataset2",
    "dataset2_cost0", "dataset3", "dataset3_cost0",
)
MODELS = ("mlp", "gnn")


def _model_paths(values: list[str], option: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{option} requires mlp=PATH or gnn=PATH")
        label, raw_path = value.split("=", 1)
        label = label.lower()
        if label not in MODELS or not raw_path or label in result:
            raise ValueError(f"{option} needs one non-empty path per selected mlp/gnn model")
        result[label] = Path(raw_path).expanduser().resolve()
    return result


def _static_source(
    label: str, *, dataset: str, group: str, seed: int, output_root: Path,
    initialize_from: dict[str, Path], static_runs: dict[str, Path],
) -> Path:
    if label in initialize_from and label in static_runs:
        raise ValueError(f"{label}: choose either --initialize-from or --static-run")
    if label in initialize_from:
        source = initialize_from[label]
    elif label in static_runs:
        source = static_runs[label] / "checkpoints" / "best.pt"
    else:
        parent = output_root / dataset / group / "static" / label / f"seed{seed}"
        candidates = sorted(parent.glob("*/checkpoints/best.pt"))
        if not candidates:
            raise FileNotFoundError(
                f"{label}: no static best.pt under {parent}; train the matching static "
                f"group first, or pass --initialize-from {label}=PATH"
            )
        if len(candidates) != 1:
            raise ValueError(
                f"{label}: found {len(candidates)} static checkpoints under {parent}; "
                f"choose explicitly with --initialize-from {label}=PATH"
            )
        source = candidates[0]
    if not source.is_file():
        raise FileNotFoundError(f"{label}: static checkpoint does not exist: {source}")
    return source.resolve()


def build_run_configs(
    *, group: str, phase: str, model: str = "both", dataset: str = "dataset2",
    seed: int = 7, world_size: int = 8, device: str = "cuda",
    output_root: str | Path = ROOT / "outputs/state_ablation",
    initialize_from: dict[str, Path] | None = None,
    static_runs: dict[str, Path] | None = None,
    config_root: str | Path = ROOT / "configs/state_ablation",
    project_root: str | Path = ROOT,
) -> dict[str, dict[str, Any]]:
    """Resolve independent run configurations; never mutate template files."""

    if group not in ("base", "full") or phase not in ("static", "dynamic"):
        raise ValueError("group must be base/full and phase must be static/dynamic")
    if model not in (*MODELS, "both") or dataset not in DATASETS:
        raise ValueError("unknown model or dataset family")
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size <= 0:
        raise ValueError("world_size must be a positive integer")
    selected = MODELS if model == "both" else (model,)
    initialization = {key: Path(value).expanduser().resolve() for key, value in (initialize_from or {}).items()}
    runs = {key: Path(value).expanduser().resolve() for key, value in (static_runs or {}).items()}
    if (set(initialization) | set(runs)) - set(selected):
        raise ValueError("checkpoint paths may only be specified for selected models")
    if phase == "static" and (initialization or runs):
        raise ValueError("static state experiments train from scratch; checkpoint arguments are dynamic-only")

    output_root = Path(output_root).expanduser().resolve()
    project_root = Path(project_root).resolve()
    configs = {}
    for label in selected:
        config = load_run_config(Path(config_root) / f"{group}_{label}_{phase}.json")
        config.pop("initialize_from", None)
        config.update(
            train_instances=str(project_root / "data" / dataset / phase / "train"),
            eval_instances=str(project_root / "data" / dataset / phase / "validation"),
            test_instances=str(project_root / "data" / dataset / phase / "test"),
            output_dir=str(output_root / dataset / group / phase / label / f"seed{seed}"),
            output_dir_mode="unique", seed=seed, distributed_world_size=world_size,
            device=device,
        )
        if phase == "dynamic":
            config["initialize_from"] = str(_static_source(
                label, dataset=dataset, group=group, seed=seed, output_root=output_root,
                initialize_from=initialization, static_runs=runs,
            ))
        configs[label] = config
    return configs


def audit_run_configs(configs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Audit only selected models; a single-model run needs only its source."""

    if set(configs) == set(MODELS):
        return audit_comparison(configs["mlp"], configs["gnn"])
    errors = []
    models = {}
    datasets = {}
    effective = {}
    for label, raw in configs.items():
        config = effective_config(raw)
        effective[label] = config
        constraint_errors = _constraints(config)
        errors.extend(f"{label}: {error}" for error in constraint_errors)
        if config["model_type"] != {"mlp": "mlp", "gnn": "gnn_attention"}[label]:
            errors.append(f"{label}: model_type does not match selected model")
        try:
            datasets[label], dataset_errors = _dataset_summary(config)
            errors.extend(f"{label}: {error}" for error in dataset_errors)
            if not constraint_errors:
                models[label] = _model_summary(config)
        except (OSError, ValueError, TypeError, KeyError, RuntimeError, EOFError) as exc:
            errors.append(f"{label}: {exc}")
    return {"passed": not errors, "errors": errors, "models": models,
            "datasets": datasets, "effective_configs": effective}


def training_command(config_path: str | Path, world_size: int) -> list[str]:
    return [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
            f"--nproc_per_node={world_size}", "-m", "ppo_rs.train", "--config", str(config_path)]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", required=True, choices=("base", "full"))
    parser.add_argument("--phase", required=True, choices=("static", "dynamic"))
    parser.add_argument("--model", default="both", choices=(*MODELS, "both"))
    parser.add_argument("--dataset", default="dataset2", choices=DATASETS)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", default=str(ROOT / "outputs/state_ablation"))
    parser.add_argument("--initialize-from", action="append", default=[], metavar="MODEL=CHECKPOINT")
    parser.add_argument("--static-run", action="append", default=[], metavar="MODEL=RUN_DIRECTORY")
    parser.add_argument("--check", action="store_true", help="CPU-only preflight; do not launch training")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        configs = build_run_configs(
            group=args.group, phase=args.phase, model=args.model, dataset=args.dataset,
            seed=args.seed, world_size=args.world_size, device=args.device,
            output_root=args.output_root,
            initialize_from=_model_paths(args.initialize_from, "--initialize-from"),
            static_runs=_model_paths(args.static_run, "--static-run"),
        )
        report = audit_run_configs(configs)
        report.update(group=args.group, phase=args.phase, dataset=args.dataset,
                      selected_models=list(configs), check_only=args.check)
        if not report["passed"] or args.check:
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if report["passed"] else 1

        device = torch.device(args.device)
        if device.type == "cuda":
            if not torch.cuda.is_available() or torch.cuda.device_count() < args.world_size:
                raise RuntimeError(f"training needs {args.world_size} visible CUDA GPUs")
            if device.index is not None:
                raise ValueError("use --device cuda and select physical GPUs with CUDA_VISIBLE_DEVICES")
        elif device.type != "cpu":
            raise ValueError("state launcher supports --device cpu or cuda")

        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        launch_dir = Path(args.output_root).expanduser().resolve() / "launches" / f"{stamp}_{uuid4().hex[:8]}"
        launch_dir.mkdir(parents=True, exist_ok=False)
        write_json(launch_dir / "preflight.json", report)
        config_paths = {}
        for label, config in configs.items():
            path = launch_dir / f"{label}_config.json"
            write_json(path, config)
            config_paths[label] = path
        state = {"status": "ready", "models": list(configs), "commands": {
            label: training_command(path, args.world_size) for label, path in config_paths.items()
        }}
        write_json(launch_dir / "launch.json", state)
        print(f"Launch records: {launch_dir}", flush=True)
        env = dict(os.environ)
        env.setdefault("OMP_NUM_THREADS", "2")
        env.setdefault("MKL_NUM_THREADS", "2")
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
        for label, path in config_paths.items():
            state.update(status="training", active_model=label)
            write_json(launch_dir / "launch.json", state)
            log_path = launch_dir / f"{label}.log"
            with log_path.open("x", encoding="utf-8") as log:
                code = subprocess.run(training_command(path, args.world_size), cwd=ROOT, env=env,
                                      stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT).returncode
            if code:
                state.update(status="failed", exit_code=code)
                write_json(launch_dir / "launch.json", state)
                raise RuntimeError(f"{label} training failed with exit {code}; see {log_path}")
            print(f"{label} complete; log: {log_path}", flush=True)
        state.update(status="complete", active_model=None)
        write_json(launch_dir / "launch.json", state)
        return 0
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, EOFError) as exc:
        print(json.dumps({"passed": False, "errors": [str(exc)]}, ensure_ascii=False, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
