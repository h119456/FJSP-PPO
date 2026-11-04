"""Run the fixed dataset2 GNN/MLP pair sequentially on all eight GPUs."""
from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]


def now() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def main() -> None:
    os.chdir(ROOT)
    configs = {label: json.loads((ROOT / source).read_text()) for label, source in (
        ("gnn", "configs/dataset2_gnn.json"), ("mlp", "configs/dataset2.json"))}
    for config in configs.values():
        directory = Path(config["output_dir"])
        if (directory / "checkpoints").exists() or (directory / "training.log").exists():
            raise RuntimeError(f"Refusing to overwrite an existing training run: {directory}")
        directory.mkdir(parents=True, exist_ok=True)
    from check_comparison import audit_comparison
    audit = audit_comparison(configs["mlp"], configs["gnn"])
    if not audit["passed"]:
        raise RuntimeError(str(audit["errors"]))
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
        raise RuntimeError("Expected eight visible CUDA GPUs")
    state = {
        "status": "training", "started_at": now(), "supervisor_pid": os.getpid(),
        "gpu_ids": list(range(8)), "order": ["gnn", "mlp"], "models": {},
    }
    for label, config in configs.items():
        directory = Path(config["output_dir"])
        (directory / "training_config.json").write_text(json.dumps(config, indent=2) + "\n")
        (directory / "preflight.json").write_text(json.dumps(audit, indent=2) + "\n")
        state["models"][label] = {"status": "queued", "directory": str(directory)}

    def save() -> None:
        state["updated_at"] = now()
        for config in configs.values():
            destination = Path(config["output_dir"]) / "status.json"
            temporary = destination.with_suffix(".tmp")
            temporary.write_text(json.dumps(state, indent=2) + "\n")
            temporary.replace(destination)

    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
               PYTHONUNBUFFERED="1")
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    process = None
    try:
        save()
        for label, config in configs.items():
            entry = state["models"][label]
            directory = Path(config["output_dir"])
            entry.update(status="training", started_at=now())
            state["active_model"] = label
            with (directory / "training.log").open("x") as log:
                process = subprocess.Popen([
                    sys.executable, "-m", "torch.distributed.run", "--standalone",
                    "--nnodes=1", "--nproc_per_node=8", "-m", "ppo_rs.train",
                    "--config", str(directory / "training_config.json"),
                ], cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log,
                    stderr=subprocess.STDOUT)
                entry["pid"] = process.pid
                save()
                code = process.wait()
            if code:
                raise RuntimeError(f"{label} training failed (exit {code}); see {directory}/training.log")
            history = json.loads((directory / "training_history.json").read_text())
            if len(history) != config["episodes"] or int(history[-1]["episode"]) != config["episodes"]:
                raise RuntimeError(f"{label}: incomplete episode history")
            if not all((directory / "checkpoints" / f"{name}.pt").is_file()
                       for name in ("best", "latest", "final")):
                raise RuntimeError(f"{label}: missing checkpoint")
            entry.update(status="complete", completed_at=now(), episodes=len(history),
                         environment_steps=history[-1]["total_environment_steps"])
            save()
        state.update(status="complete", active_model=None, completed_at=now())
        save()
    except BaseException as exc:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait()
        active = state.get("active_model")
        if active:
            state["models"][active]["status"] = "failed"
        state.update(status="failed", error=str(exc), failed_at=now())
        save()
        raise


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise SystemExit(1)
