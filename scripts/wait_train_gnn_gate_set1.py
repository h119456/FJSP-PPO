"""Wait for the dataset1 chain and idle GPUs 0-1, then run GNN+gate."""

from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
JOBS = ("static/MLP", "static/GNN", "dynamic/MLP", "dynamic/GNN")


def now() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def gpu_processes(gpu_ids=(0, 1)) -> list[str]:
    """Use GPU UUIDs: nvidia-smi PIDs can belong to the host PID namespace."""
    devices = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        text=True, timeout=20,
    )
    mapping = dict(line.strip().split(", ", 1) for line in devices.splitlines() if line.strip())
    selected = {mapping[str(index)] for index in gpu_ids}
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        text=True, timeout=20,
    )
    return [line.strip() for line in apps.splitlines()
            if line.strip() and line.split(",", 1)[0].strip() in selected]


def readiness(root: Path) -> tuple[bool, str]:
    target = root / "outputs/GNN+gate_set1"
    if target.exists():
        raise RuntimeError(f"Target already exists; refusing another run: {target}")
    source = root / "outputs/simplestate_jingtai_set1/pipeline_status.json"
    if not source.is_file():
        return False, "Waiting for simplestate dataset1 pipeline status"
    status = json.loads(source.read_text())
    if status["status"] in {"failed", "interrupted", "error", "cancelled", "stopped"}:
        raise RuntimeError(f"Previous pipeline stopped: {status['status']}")
    if status["status"] != "complete":
        return False, f"Waiting for previous pipeline: {status.get('active_job', status['status'])}"
    if not all(status.get("jobs", {}).get(job, {}).get("status") == "complete" for job in JOBS):
        raise RuntimeError("Previous pipeline says complete but its four jobs are incomplete")
    processes = gpu_processes()
    if processes:
        return False, f"Waiting for GPU0-1 compute processes: {processes}"
    return True, "Previous four jobs complete; GPUs 0-1 have no compute processes"


def write_status(path: Path, state: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(dict(state, updated_at=now()), ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def run(root: Path, interval: int) -> int:
    outputs = root / "outputs"
    outputs.mkdir(exist_ok=True)
    status_path = outputs / "gnn_gate_set1_waiter_status.json"
    with (outputs / ".gnn_gate_set1_waiter.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Another GNN+gate dataset1 waiter is already running", flush=True)
            return 1
        state = dict(status="waiting", pid=os.getpid(), started_at=now(), gpu_ids=[0, 1],
                     output_dir=str(outputs / "GNN+gate_set1"), poll_seconds=interval)
        child = None

        def stop(signum, _frame):
            raise InterruptedError(f"Received signal {signum}")

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        consecutive_idle = 0
        last_reason = None
        try:
            while True:
                try:
                    ready, reason = readiness(root)
                except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
                    # An unreadable status or failed GPU query is never permission to launch.
                    ready, reason = False, f"Cannot establish readiness: {exc}"
                consecutive_idle = consecutive_idle + 1 if ready else 0
                state.update(status="waiting", reason=reason, consecutive_idle=consecutive_idle)
                write_status(status_path, state)
                if reason != last_reason:
                    print(f"{now()} {reason}", flush=True)
                    last_reason = reason
                if consecutive_idle >= 2:
                    break
                time.sleep(interval)
            command = ["bash", str(root / "scripts/train_gnn_gate_set1.sh")]
            child = subprocess.Popen(command, cwd=root, start_new_session=True)
            state.update(status="training", child_pid=child.pid, command=command, launched_at=now())
            write_status(status_path, state)
            print(f"{now()} Starting static then dynamic GNN+gate; child PID={child.pid}", flush=True)
            code = child.wait()
            state.update(status="complete" if code == 0 else "failed", returncode=code, completed_at=now())
            write_status(status_path, state)
            return code
        except (InterruptedError, KeyboardInterrupt) as exc:
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
            state.update(status="interrupted", reason=str(exc))
            write_status(status_path, state)
            return 130
        except Exception as exc:
            state.update(status="failed", reason=str(exc))
            write_status(status_path, state)
            print(f"{now()} Failed: {exc}", flush=True)
            return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--check", action="store_true", help="Print readiness without creating files or launching")
    args = parser.parse_args()
    if not 1 <= args.poll_seconds <= 60:
        parser.error("--poll-seconds must be between 1 and 60")
    if args.check:
        ready, reason = readiness(ROOT)
        print(json.dumps(dict(ready=ready, reason=reason), ensure_ascii=False))
        return 0
    return run(ROOT, args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
