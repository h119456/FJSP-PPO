"""在 GPU0/1 上对已结束实验的验证集最佳模型进行独立测试集评估。"""
from __future__ import annotations

import argparse
import csv
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
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
GPUS = (0, 1)
SEED = 1007


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def planned_runs(root=ROOT):
    runs = []
    for state in ("fullstate", "simplestate"):
        for dataset in (1, 2, 3):
            for phase, label in (("static", "jingtai"), ("dynamic", "dongtai")):
                for model in ("MLP", "GNN"):
                    runs.append(dict(id=f"{state}_set{dataset}_{phase}_{model}", dataset=dataset,
                                     phase=phase, state=state, model=model,
                                     run=str(root / f"outputs/{state}_{label}_set{dataset}/{model}")))
    for variant, datasets in (("gate", (1, 2, 3)), ("FiLM", (1,))):
        for dataset in datasets:
            for phase in ("static", "dynamic"):
                runs.append(dict(id=f"GNN_{variant}_set{dataset}_{phase}", dataset=dataset,
                                 phase=phase, state="fullstate", model=f"GNN+{variant}",
                                 run=str(root / f"outputs/GNN+{variant}_set{dataset}/{phase}")))
    return runs


def training_outcome(job):
    run = Path(job["run"])
    if not (run / "training_config.json").exists():
        return "not_started"
    if (run / "checkpoints/final.pt").exists() and (run / "checkpoint_summary.json").exists():
        return "complete"
    if (run / "early_stop.json").exists():
        return "early_stopped"
    independent = run / "independent_status.json"
    if independent.exists():
        return read(independent)["status"]
    status = run.parent / "pipeline_status.json"
    if status.exists():
        state = read(status)
        key = job["phase"] if job["model"] not in ("MLP", "GNN") else f"{job['phase']}/{job['model']}"
        return state.get("jobs", {}).get(key, {}).get("status", "queued")
    return "queued"


def successful(metrics):
    return (bool(metrics["terminated"]) and not bool(metrics["truncated"])
            and int(metrics["schedule_violation_count"]) == 0
            and int(metrics["process_readiness_violations"]) == 0)


def summarize(rows):
    from scripts.evaluate_dataset import _aggregate
    result = _aggregate(rows)
    valid = [row["objective"] for row in rows if successful(row)]
    result.update(valid_instance_count=len(valid), completion_rate=len(valid) / len(rows),
                  successful_mean_objective=sum(valid) / len(valid) if valid else None,
                  truncated_instance_count=sum(bool(row["truncated"]) for row in rows))
    return result


def evaluation_source(job, config, root=ROOT):
    """Allow an explicit static-to-dynamic test without changing training files."""
    evaluation_phase = job.get('evaluation_phase', job.get('phase'))
    if evaluation_phase == job.get('phase'):
        return Path(config['test_instances']).resolve()
    if (job.get('phase'), evaluation_phase) != ('static', 'dynamic'):
        raise RuntimeError('Only an explicit static-to-dynamic cross-phase test is supported')
    return (root / f"data/dataset{job['dataset']}_cost0/dynamic/test").resolve()


FAULT_SCENARIOS = ('breakdown_only', 'both')


def evaluation_indices(job, paths):
    if not job.get('fault_only', False):
        return list(range(len(paths)))
    indices = [index for index, path in enumerate(paths)
               if read(path).get('dynamic_events', {}).get('scenario') in FAULT_SCENARIOS]
    if len(indices) != 30:
        raise RuntimeError(f'Expected exactly 30 fault scenarios, got {len(indices)}')
    return indices


def cached_evaluation(path, *, checkpoint_hash, source, manifest, profiles, names, scenarios, indices):
    """Validate cached rows against their original full-test positions and seeds."""
    try:
        old = read(path)
        if (old.get('checkpoint_sha256') != checkpoint_hash or old.get('seed') != SEED
                or old.get('deterministic') is not True or old.get('test_manifest') != manifest
                or old.get('test_instances') != str(source)
                or any(old.get(key) != value for key, value in profiles.items())):
            return None
        lookup = {row['name']: row for row in old['instances']}
        if len(lookup) != len(old['instances']):
            return None
        rows = [lookup[names[index]] for index in indices]
        for index, row in zip(indices, rows):
            metrics = row['metrics']
            if (row['seed'] != SEED + index or row['scenario'] != scenarios[index]
                    or not math.isfinite(float(metrics['objective']))):
                return None
            successful(metrics)  # Require all completion/violation fields.
        return old, rows
    except (OSError, ValueError, TypeError, KeyError):
        return None


def evaluate_job(job_path, destination):
    import torch
    from ppo_rs import load_instances, resolve_instance_paths
    from ppo_rs.cli_utils import set_seed
    from ppo_rs.environment import ResidualStressFJSPEnv
    from ppo_rs.environment.checkpoint_semantics import check_dynamic_input_semantics
    from ppo_rs.evaluation import evaluate_model
    from scripts.evaluate_dataset import _load_model, _checkpoint_profiles
    job = read(job_path)
    run = Path(job["run"])
    config = read(run / "training_config.json")
    evaluation_phase = job.get('evaluation_phase', job['phase'])
    expected = ROOT / f"data/dataset{job['dataset']}_cost0/{evaluation_phase}/test"
    source = evaluation_source(job, config)
    if source != expected or source in (Path(config["train_instances"]), Path(config["eval_instances"])):
        raise RuntimeError("Expected an independent matching test split")
    checkpoint_path = run / "checkpoints/best.pt"
    checkpoint_hash = digest(checkpoint_path)
    paths = resolve_instance_paths(source)
    if len(paths) != 200:
        raise RuntimeError(f"Expected 200 test instances, got {len(paths)}")
    manifest = [{"name": path.name, "sha256": digest(path)} for path in paths]
    indices = evaluation_indices(job, paths)
    torch.set_num_threads(2)
    set_seed(SEED)
    model, checkpoint = _load_model(checkpoint_path, torch.device("cuda:0"))
    profiles = _checkpoint_profiles(checkpoint)
    instances = load_instances(source)
    names = [instance.name for instance in instances]
    scenario_names = [read(path).get('dynamic_events', {}).get('scenario', 'none') for path in paths]
    check_dynamic_input_semantics(checkpoint, dynamic=any(i.dynamic_events for i in instances))
    vals = read(run / "validation_history.json")
    eligible = [v for v in vals if v["terminated"] and not v["truncated"]]
    best = min(eligible, key=lambda v: v["objective"])
    if (int(checkpoint["episode"]) != best["episode"]
            or not math.isclose(best["objective"], checkpoint["best_validation_objective"], abs_tol=1e-8)):
        raise RuntimeError("Checkpoint disagrees with the validation selection record")
    metadata = dict(job=job, checkpoint=str(checkpoint_path), checkpoint_sha256=checkpoint_hash,
                    checkpoint_episode=int(checkpoint["episode"]),
                    best_validation_objective=best["objective"], seed=SEED, deterministic=True,
                    device="cuda:0", physical_gpu=int(os.environ["CUDA_VISIBLE_DEVICES"]),
                    test_instances=str(source), test_manifest=manifest, **profiles,
                    parameter_count=sum(p.numel() for p in model.parameters()), started_at=now())
    metadata.update(checkpoint_training_phase=job['phase'], evaluation_phase=evaluation_phase)
    metadata.update(evaluated_indices=indices, fault_only=bool(job.get('fault_only', False)))
    for previous in sorted(run.glob("test_evaluation_*/results.json")):
        cached = cached_evaluation(previous, checkpoint_hash=checkpoint_hash, source=source,
                                   manifest=manifest, profiles=profiles, names=names,
                                   scenarios=scenario_names, indices=indices)
        if cached is not None:
            _, rows = cached
            metadata['reused_from'] = str(previous)
            break
    else:
        rows = []
    if not rows:
        progress_path = destination.with_suffix(".progress.json")
        rows_path = destination.with_suffix(".instances.jsonl")
        with rows_path.open("x") as stream:
            for completed, index in enumerate(indices, start=1):
                instance, path = instances[index], paths[index]
                metrics, _ = evaluate_model(ResidualStressFJSPEnv(instance, **profiles), model,
                                            seed=SEED + index, deterministic=True)
                if not math.isfinite(float(metrics["objective"])):
                    raise RuntimeError(f"Nonfinite objective: {instance.name}")
                scenario = read(path).get("dynamic_events", {}).get("scenario", "none")
                row = dict(name=instance.name, scenario=scenario, seed=SEED + index, metrics=metrics)
                rows.append(row)
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                if completed % 10 == 0 or completed == len(indices):
                    write(progress_path, dict(completed=completed, total=len(indices), updated_at=now()))
                    print(f"{now()} {job['id']}: {completed}/{len(indices)}", flush=True)
    if digest(checkpoint_path) != checkpoint_hash or any(digest(p) != m["sha256"] for p, m in zip(paths, manifest)):
        raise RuntimeError("Checkpoint or test data changed during evaluation")
    aggregate = summarize([r["metrics"] for r in rows])
    scenarios = {s: summarize([r["metrics"] for r in rows if r["scenario"] == s])
                 for s in sorted({r["scenario"] for r in rows})}
    write(destination, dict(**metadata, completed_at=now(), aggregate=aggregate,
                            scenario_aggregates=scenarios, instances=rows))
    print(json.dumps(dict(job=job["id"], aggregate=aggregate), ensure_ascii=False), flush=True)


def export_summary(output, state):
    fields = ["id", "dataset", "phase", "state", "model", "training_status", "status",
              "best_validation", "checkpoint_episode", "successful_mean_objective", "completion_rate",
              "mean_objective", "valid_instance_count", "instance_count", "result"]
    rows = []
    for job in state["jobs"]:
        row = {key: job.get(key, "") for key in fields}
        if job["status"] == "complete":
            result = read(Path(job["result"]))
            row.update({k: result["aggregate"].get(k, "") for k in fields if k in result["aggregate"]})
            row.update(best_validation=result["best_validation_objective"],
                       checkpoint_episode=result["checkpoint_episode"])
        rows.append(row)
    tmp = output / "summary.csv.tmp"
    with tmp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(output / "summary.csv")
    phase_description = (
        "跨阶段对照：静态验证集选择静态best.pt，在对应动态测试集评估；训练配置保持原样。"
        if any(j.get('evaluation_phase', j['phase']) != j['phase'] for j in state['jobs'])
        else "静态与动态使用各自测试集；训练未结束的实验不评估中途best。")
    lines = ["# 独立测试集评估", "", "验证集选择best.pt；测试集固定200例，seed1007，确定性动作。",
             "主列为成功实例平均目标，同时报告完成率；全量均值保留失败惩罚，不按目标大小删除样本。",
             phase_description, "",
             "| 实验 | 训练状态 | 评估状态 | 成功实例均值 | 有效完成 | 全量均值 |",
             "|---|---|---|---:|---:|---:|"]
    for row in rows:
        if row["status"] == "complete":
            value = row['successful_mean_objective']
            display = f"{value:.6f}" if value is not None else "空"
            lines.append(f"| {row['id']} | {row['training_status']} | complete | {display}"
                         f" | {row['valid_instance_count']}/{row['instance_count']} | {row['mean_objective']:.6f} |")
        else:
            lines.append(f"| {row['id']} | {row['training_status']} | {row['status']} | | | |")
    (output / "summary.md").write_text("\n".join(lines) + "\n")


def run_suite(output):
    from scripts.wait_train_gnn_gate_set1 import gpu_processes
    with (ROOT / "outputs/.test_evaluation_suite.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any(gpu_processes((gpu,)) for gpu in GPUS):
            raise RuntimeError("GPUs 0/1 are occupied")
        output.mkdir(exist_ok=False)
        (output / "results").mkdir()
        (output / "logs").mkdir()
        jobs = [dict(job, status="pending") for job in planned_runs()]
        state = dict(status="running", started_at=now(), supervisor_pid=os.getpid(),
                     gpu_ids=list(GPUS), seed=SEED, jobs=jobs)
        running = {}

        def save():
            state["updated_at"] = now()
            state["counts"] = {s: sum(j["status"] == s for j in jobs) for s in sorted({j['status'] for j in jobs})}
            write(output / "status.json", state)
            export_summary(output, state)

        def stop(signum, _frame):
            raise InterruptedError(f"Received signal {signum}")

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        try:
            while True:
                for gpu, (process, job, log) in list(running.items()):
                    code = process.poll()
                    if code is not None:
                        log.close()
                        del running[gpu]
                        job.update(status="complete" if code == 0 and Path(job["result"]).exists() else "failed",
                                   returncode=code, ended_at=now())
                for job in jobs:
                    if job["status"] not in ("complete", "failed", "evaluating"):
                        outcome = training_outcome(job)
                        job["training_status"] = outcome
                        best = Path(job["run"]) / "checkpoints/best.pt"
                        job["status"] = ("ready" if outcome in ("complete", "early_stopped", "failed") and best.exists()
                                         else "unavailable" if outcome in ("failed", "stopped", "interrupted")
                                         else "not_started" if outcome == "not_started" else "waiting_training")
                for gpu in GPUS:
                    if gpu in running or gpu_processes((gpu,)):
                        continue
                    ready = next((j for j in jobs if j["status"] == "ready"), None)
                    if ready is None:
                        continue
                    job_path = output / "results" / (ready["id"] + ".job.json")
                    result = output / "results" / (ready["id"] + ".json")
                    write(job_path, ready)
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="2",
                               MKL_NUM_THREADS="2", PYTHONUNBUFFERED="1", PYTHONPATH=str(ROOT / "src") + ":" + str(ROOT))
                    command = [sys.executable, "-u", str(Path(__file__).resolve()),
                               "--worker", str(job_path), "--result", str(result)]
                    log = (output / "logs" / (ready["id"] + ".log")).open("x")
                    process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    ready.update(status="evaluating", gpu=gpu, pid=process.pid,
                                 started_at=now(), result=str(result))
                    running[gpu] = process, ready, log
                    print(f"{now()} GPU{gpu}: {ready['id']}", flush=True)
                save()
                if not running and not any(j["status"] in ("ready", "waiting_training", "pending") for j in jobs):
                    state.update(status="complete_with_unavailable" if any(j['status'] != 'complete' for j in jobs)
                                 else "complete", completed_at=now())
                    save()
                    return
                time.sleep(15)
        except BaseException as exc:
            for process, job, log in running.values():
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                process.wait()
                log.close()
                job["status"] = "interrupted"
            state.update(status="interrupted" if isinstance(exc, (InterruptedError, KeyboardInterrupt))
                         else "failed", error=str(exc))
            save()
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.worker:
        if args.result is None:
            parser.error("--worker requires --result")
        evaluate_job(args.worker, args.result)
    elif args.check:
        print(json.dumps([dict(j, training_status=training_outcome(j)) for j in planned_runs()], indent=2))
    elif args.output is None:
        parser.error("--output is required")
    else:
        run_suite(args.output.resolve())
