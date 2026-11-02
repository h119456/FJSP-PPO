"""Optional Matplotlib visualizations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence


COLORS = {
    "NORMAL": "#4C78A8",
    "STRESS_RELIEF": "#F58518",
    "COMPENSATION": "#E45756",
}


def _pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency message
        raise RuntimeError("plotting requires matplotlib; install requirements.txt") from exc
    return plt


def plot_gantt(schedule: Sequence[dict[str, Any]], output: str | Path) -> None:
    plt = _pyplot()
    machines = sorted({str(entry["machine_id"]) for entry in schedule})
    positions = {machine: index for index, machine in enumerate(machines)}
    figure, axis = plt.subplots(figsize=(10, max(3, 0.7 * len(machines))))
    for entry in schedule:
        start = float(entry["start"])
        duration = float(entry["end"]) - start
        kind = str(entry["kind"])
        y = positions[str(entry["machine_id"])]
        axis.barh(y, duration, left=start, height=0.55, color=COLORS.get(kind, "#999999"))
        axis.text(
            start + duration / 2,
            y,
            f"{entry['job_id']}:{entry['operation_id']}",
            ha="center",
            va="center",
            fontsize=8,
            color="white",
        )
    axis.set_yticks(range(len(machines)), machines)
    axis.set_xlabel("Time")
    axis.set_ylabel("Machine")
    axis.set_title("Residual-stress-aware FJSP schedule")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=160)
    plt.close(figure)


def plot_training_curve(history: Sequence[dict[str, float]], output: str | Path) -> None:
    plt = _pyplot()
    episodes = [int(row["episode"]) for row in history]
    objectives = [float(row["objective"]) for row in history]
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(episodes, objectives, color="#4C78A8", linewidth=1.5)
    axis.set_xlabel("Episode")
    axis.set_ylabel("Objective J")
    axis.set_title("PPO training objective")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=160)
    plt.close(figure)

