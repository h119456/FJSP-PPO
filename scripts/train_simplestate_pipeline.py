"""Run each cost0 dataset's simplified-state static pair, then dynamic pair.

--check is CPU-only and creates no output directories. Without --check this
is a foreground supervisor; use tmux or nohup for a persistent manual launch.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.check_comparison import (  # noqa: E402
    ARCHITECTURE_KEYS, _constraints, _dataset_summary, _model_summary, effective_config,
)
from scripts.make_zero_cost_dataset import validate_dataset as audit_cost0  # noqa: E402
from scripts.validate_dynamic_dataset import validate_dataset as audit_dynamic  # noqa: E402
from scripts.train_state_comparison import training_command  # noqa: E402

GPU_GROUPS = {1: [0, 1], 2: [2, 3, 4], 3: [5, 6, 7]}
ORDER = [(phase, model) for phase in ("static", "dynamic") for model in ("MLP", "GNN")]
STOP = None
CHILDREN = []


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stop(number, _frame):
    global STOP
    STOP = number
    for process in CHILDREN:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass


def phase_dirs(number, output_root):
    return {"static": output_root / f"simplestate_jingtai_set{number}",
            "dynamic": output_root / f"simplestate_dongtai_set{number}"}


def build_configs(number, output_root, seed=7):
    directories = phase_dirs(number, output_root)
    result = {}
    for phase, model in ORDER:
        config = json.loads((ROOT / f"configs/state_ablation/base_{model.lower()}_{phase}.json").read_text())
        config.update(seed=seed, eval_seed=1007, distributed_world_size=len(GPU_GROUPS[number]),
                      output_dir=str(directories[phase] / model), output_dir_mode="existing")
        for key, split in (("train_instances", "train"), ("eval_instances", "validation"),
                           ("test_instances", "test")):
            config[key] = str(ROOT / f"data/dataset{number}_cost0/{phase}/{split}")
        if phase == "dynamic":
            config["initialize_from"] = str(directories["static"] / model / "checkpoints/best.pt")
        else:
            config.pop("initialize_from", None)
        result.setdefault(phase, {})[model.lower()] = config
    return result


def audit_pair(configs, pending_sources=False):
    """Use existing schema checks while supporting PPO's masked uneven batches.

    The legacy comparison checker requires divisibility, although PPO already
    masks padding. Keep that checker unchanged: running full-state supervisors
    pin its hash. Validate its other constraints with a divisible batch, then
    inspect the real, unchanged batch/model/dataset settings below.
    """
    effective = {k: effective_config(v) for k, v in configs.items()}
    differences = {key for key in set(effective["mlp"]) | set(effective["gnn"])
                   if key not in ARCHITECTURE_KEYS and effective["mlp"].get(key) != effective["gnn"].get(key)}
    errors = [f"training settings differ: {sorted(differences)}"] if differences else []
    models, datasets = {}, {}
    for label, config in effective.items():
        inspected = dict(config)
        batch, world = config["mini_batch_size"], config["distributed_world_size"]
        if isinstance(batch, int) and not isinstance(batch, bool) and batch > 0 and isinstance(world, int) and not isinstance(world, bool) and world > 0:
            inspected["mini_batch_size"] = math.ceil(batch / world) * world
        constraint_errors = _constraints(inspected)
        errors.extend(f"{label}: {e}" for e in constraint_errors)
        if config["model_type"] != {"mlp": "mlp", "gnn": "gnn_attention"}[label]:
            errors.append(f"{label}: incorrect model_type")
        if config["observation_profile"] != "compact18" or config["action_feature_profile"] != "compact21":
            errors.append(f"{label}: simplified state requires compact18/compact21")
        try:
            datasets[label], dataset_errors = _dataset_summary(config)
            errors.extend(f"{label}: {e}" for e in dataset_errors)
            if constraint_errors:
                continue
            model_config = dict(config)
            if pending_sources:
                model_config.pop("initialize_from", None)
            model = _model_summary(model_config)
            model.update(local_mini_batch_size=math.ceil(batch / world),
                         padded_global_mini_batch_size=math.ceil(batch / world) * world,
                         full_batch_valid_samples_per_rank=[max(0, min(math.ceil(batch / world), batch - rank * math.ceil(batch / world))) for rank in range(world)],
                         padding_contributes_to_loss=False)
            models[label] = model
        except (OSError, ValueError, TypeError, KeyError, RuntimeError, EOFError) as exc:
            errors.append(f"{label}: {exc}")
    return {"passed": not errors, "errors": errors, "models": models, "datasets": datasets,
            "effective_configs": effective, "setting_differences": sorted(differences),
            "source_compatibility": "pending_static_completion" if pending_sources else "checked_if_present"}


def preflight(number, configs):
    data = {"cost0": audit_cost0(ROOT / f"data/dataset{number}", ROOT / f"data/dataset{number}_cost0"),
            "dynamic": audit_dynamic(ROOT / f"data/dataset{number}_cost0/dynamic")}
    phases = {phase: audit_pair(pair, pending_sources=phase == "dynamic") for phase, pair in configs.items()}
    return {"passed": all(p["passed"] for p in phases.values()), "dataset": number,
            "gpu_ids": GPU_GROUPS[number], "data": data, "phases": phases}


def require_fullstate_finished():
    for number in (2, 3):
        path = ROOT / f"outputs/fullstate_jingtai_set{number}/pipeline_status.json"
        if path.exists() and json.loads(path.read_text())["status"] != "complete":
            raise RuntimeError(f"Full-state training has not completed: {path}; retry after both chains finish")


def validate_finished(config):
    directory = Path(config["output_dir"])
    history = json.loads((directory / "training_history.json").read_text())
    if len(history) != config["episodes"] or int(history[-1]["episode"]) != config["episodes"]:
        raise RuntimeError(f"Incomplete training history: {directory}")
    for name in ("best", "latest", "final"):
        if not (directory / "checkpoints" / f"{name}.pt").is_file():
            raise RuntimeError(f"Missing {name}.pt: {directory}")
    summary = json.loads((directory / "checkpoint_summary.json").read_text())
    return {"episodes": len(history), "environment_steps": history[-1]["total_environment_steps"],
            "best_validation_objective": summary["best_validation_objective"]}


def verify_source(config, source):
    import torch
    checkpoint = torch.load(source["path"], map_location="cpu", weights_only=True)
    previous = checkpoint["config"]
    expected = str(Path(config["train_instances"]).parents[1] / "static/train")
    if previous["train_instances"] != expected or previous["seed"] != config["seed"]:
        raise RuntimeError("Static checkpoint dataset/seed differs from this dynamic run")
    return dict(source, source_config=previous, weights_only_transfer=True, optimizer_reset=True)


def run_dataset(number, output_root, seed):
    require_fullstate_finished()
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / f".simplestate_set{number}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        directories = phase_dirs(number, output_root)
        for directory in directories.values():
            if directory.exists():
                raise FileExistsError(f"Refusing to overwrite existing run: {directory}")
        configs = build_configs(number, output_root, seed)
        report = preflight(number, configs)
        if not report["passed"]:
            raise RuntimeError(json.dumps(report, ensure_ascii=False))
        if STOP is not None:
            raise InterruptedError("Stop requested during preflight")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, GPU_GROUPS[number]))
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != len(GPU_GROUPS[number]):
            raise RuntimeError(f"Expected visible GPUs {GPU_GROUPS[number]}")
        env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONUNBUFFERED="1")
        env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
        state = {"status": "training", "dataset": number, "gpu_ids": GPU_GROUPS[number],
                 "seed": seed, "started_at": now(), "supervisor_pid": os.getpid(),
                 "order": [f"{p}/{m}" for p, m in ORDER], "jobs": {}, "automatic_early_stopping": False}
        for phase, directory in directories.items():
            directory.mkdir(exist_ok=False)
            write(directory / "preflight.json", report)
            for label, config in configs[phase].items():
                model_dir = Path(config["output_dir"])
                model_dir.mkdir(exist_ok=False)
                path = model_dir / "training_config.json"
                write(path, config)
                state["jobs"][f"{phase}/{label.upper()}"] = dict(status="queued", config=str(path),
                                                               config_sha256=sha(path), episodes_target=config["episodes"])

        def save():
            state["updated_at"] = now()
            for directory in directories.values():
                write(directory / "pipeline_status.json", state)

        active = None
        try:
            save()
            for phase in ("static", "dynamic"):
                if STOP is not None:
                    raise InterruptedError("Stop requested")
                if phase == "dynamic":
                    for label in ("MLP", "GNN"):
                        validate_finished(configs["static"][label.lower()])
                    transfer = audit_pair(configs["dynamic"])
                    write(directories["dynamic"] / "transfer_preflight.json", transfer)
                    if not transfer["passed"]:
                        raise RuntimeError(f"Transfer failed: {transfer['errors']}")
                    sources = {label.upper(): verify_source(configs["dynamic"][label], value["initialization"])
                               for label, value in transfer["models"].items()}
                    write(directories["dynamic"] / "initialization_sources.json", sources)
                for label in ("MLP", "GNN"):
                    if STOP is not None:
                        raise InterruptedError("Stop requested")
                    active = f"{phase}/{label}"
                    entry = state["jobs"][active]
                    config = configs[phase][label.lower()]
                    if sha(Path(entry["config"])) != entry["config_sha256"]:
                        raise RuntimeError(f"Configuration changed: {entry['config']}")
                    if phase == "dynamic" and sha(Path(sources[label]["path"])) != sources[label]["sha256"]:
                        raise RuntimeError(f"Static source changed: {label}")
                    command = training_command(entry["config"], len(GPU_GROUPS[number]))
                    entry.update(status="training", started_at=now(), command=command)
                    state["active_job"] = active
                    save()
                    print(f"{now()} dataset{number}: starting {active}", flush=True)
                    with (Path(config["output_dir"]) / "training.log").open("x") as log:
                        process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                        CHILDREN.append(process)
                        if STOP is not None:
                            stop(STOP, None)
                        entry["pid"] = process.pid
                        save()
                        code = process.wait()
                        CHILDREN.remove(process)
                    if STOP is not None:
                        raise InterruptedError("Stop requested")
                    if code:
                        raise RuntimeError(f"{active} exited {code}; see {config['output_dir']}/training.log")
                    entry.update(status="complete", completed_at=now(), **validate_finished(config))
                    save()
                    print(f"{now()} dataset{number}: completed {active}", flush=True)
            state.update(status="complete", active_job=None, completed_at=now())
            save()
        except BaseException as exc:
            for process in list(CHILDREN):
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                process.wait()
                CHILDREN.remove(process)
            outcome = "stopped" if STOP is not None else "failed"
            if active:
                state["jobs"][active].update(status=outcome, ended_at=now())
            state.update(status=outcome, error=str(exc), ended_at=now())
            save()
            raise


def run_all(output_root, seed):
    require_fullstate_finished()
    for number in GPU_GROUPS:
        for directory in phase_dirs(number, output_root).values():
            if directory.exists():
                raise FileExistsError(f"Refusing to overwrite existing run: {directory}")
    try:
        for number in GPU_GROUPS:
            if STOP is not None:
                raise InterruptedError("Stop requested")
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--dataset", str(number),
                                        "--output-root", str(output_root), "--seed", str(seed)],
                                       cwd=ROOT, start_new_session=True)
            CHILDREN.append(process)
            if STOP is not None:
                stop(STOP, None)
        pending, failures = list(CHILDREN), []
        while pending:
            for process in pending[:]:
                code = process.poll()
                if code is not None:
                    pending.remove(process)
                    CHILDREN.remove(process)
                    if code:
                        failures.append((process.pid, code))
            if pending:
                time.sleep(1)
        if failures:
            raise RuntimeError(f"Dataset supervisors failed: {failures}; inspect per-dataset logs/status")
    finally:
        for process in list(CHILDREN):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
            process.wait()
            CHILDREN.remove(process)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("all", "1", "2", "3"), default="all")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--check", action="store_true", help="CPU-only audit; no training or output directories")
    args = parser.parse_args(argv)
    output_root = args.output_root.expanduser().resolve()
    try:
        if args.check:
            selected = GPU_GROUPS if args.dataset == "all" else [int(args.dataset)]
            reports = [preflight(n, build_configs(n, output_root, args.seed)) for n in selected]
            print(json.dumps(reports, ensure_ascii=False, indent=2))
            return 0 if all(r["passed"] for r in reports) else 1
        if args.dataset == "all":
            run_all(output_root, args.seed)
        else:
            run_dataset(int(args.dataset), output_root, args.seed)
        return 0
    except (OSError, ValueError, KeyError, RuntimeError, InterruptedError) as exc:
        print(f"{now()} ERROR: {exc}", file=sys.stderr, flush=True)
        return 128 + STOP if STOP is not None else 1


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    raise SystemExit(main())
