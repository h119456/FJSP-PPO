"""独立启动 set3 简化 state 动态 GNN，保留前序 MLP 失败记录。"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.train_simplestate_pipeline import (
    audit_pair, now, sha, validate_finished, verify_source, write,
)
from scripts.wait_train_gnn_gate_set1 import gpu_processes


def preflight():
    directory = ROOT / "outputs/simplestate_dongtai_set3/GNN"
    config_path = directory / "training_config.json"
    config = json.loads(config_path.read_text())
    expected = {
        "model_type": "gnn_attention", "observation_profile": "compact18",
        "action_feature_profile": "compact21", "episodes": 10000,
        "distributed_world_size": 3, "mini_batch_size": 64, "seed": 7,
        "output_dir": str(directory),
        "initialize_from": str(ROOT / "outputs/simplestate_jingtai_set3/GNN/checkpoints/best.pt"),
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Existing configuration differs from the authorized experiment")
    for key in ("physical_gating", "shared_kv_film"):
        if config.get(key, False):
            raise RuntimeError(f"Baseline GNN requires {key}=False")
    if {p.name for p in directory.iterdir()} != {"training_config.json"}:
        raise RuntimeError("Target contains existing run artifacts; refusing to overwrite")
    static = json.loads((ROOT / "outputs/simplestate_jingtai_set3/GNN/training_config.json").read_text())
    validate_finished(static)
    pair = {label.lower(): json.loads((directory.parent / label / "training_config.json").read_text())
            for label in ("MLP", "GNN")}
    report = audit_pair(pair)
    if not report["passed"]:
        raise RuntimeError(str(report["errors"]))
    source = verify_source(config, report["models"]["gnn"]["initialization"])
    busy = gpu_processes((5, 6, 7))
    if busy:
        raise RuntimeError(f"GPUs 5-7 occupied: {busy}")
    return directory, config_path, config, report, source


def run():
    with (ROOT / "outputs/.simplestate_set3.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        directory, config_path, config, report, source = preflight()
        write(directory / "independent_preflight.json", report)
        write(directory / "initialization_source.json", source)
        state_path = directory / "independent_status.json"
        state = dict(status="preflight", supervisor_pid=os.getpid(), gpu_ids=[5, 6, 7],
                     started_at=now(), episodes_target=config["episodes"],
                     config_sha256=sha(config_path), previous_mlp_status="failed")
        child = None

        def save():
            state["updated_at"] = now()
            write(state_path, state)

        def stop(signum, _frame):
            raise InterruptedError(f"Received signal {signum}")

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        save()
        try:
            if sha(Path(source["path"])) != source["sha256"]:
                raise RuntimeError("Static source changed after preflight")
            command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                       "--nnodes=1", "--nproc_per_node=3", "-m", "ppo_rs.train",
                       "--config", str(config_path)]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="5,6,7", OMP_NUM_THREADS="2",
                       MKL_NUM_THREADS="2", PYTHONUNBUFFERED="1",
                       PYTHONPATH=str(ROOT / "src") + os.pathsep + str(ROOT))
            with (directory / "training.log").open("x") as log:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                state.update(status="training", pid=child.pid, command=command)
                save()
                print(f"{now()} Started dynamic GNN; PID={child.pid}", flush=True)
                code = child.wait()
            if code:
                raise RuntimeError(f"Training exited {code}; see {directory / 'training.log'}")
            state.update(status="complete", completed_at=now(), **validate_finished(config))
            save()
            return 0
        except BaseException as exc:
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait()
            state.update(status="interrupted" if isinstance(exc, (InterruptedError, KeyboardInterrupt))
                         else "failed", error=str(exc), ended_at=now())
            save()
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        _, _, _, _, source = preflight()
        print(json.dumps(dict(passed=True, gpu_ids=[5, 6, 7], source=source), ensure_ascii=False))
    else:
        raise SystemExit(run())
