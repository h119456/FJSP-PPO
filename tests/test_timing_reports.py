from __future__ import annotations

import pytest

from scripts.compare_observation_runs import _summarize_rows
from scripts.evaluate_dataset import _aggregate


def _rows(*, timed: bool) -> list[dict]:
    rows = []
    for index in range(2):
        row = {
            "objective": 8.0,
            "makespan": 7.0,
            "normal_processing_time": 4.0,
            "inter_operation_idle_time": 1.0,
            "stress_relief_count": 1,
            "compensation_count": 1,
            "total_independent_cost": 2.0,
            "terminated": True,
            "truncated": False,
            "schedule_violation_count": 0,
            "process_readiness_violations": 0,
        }
        if timed:
            row.update(
                episode_wall_seconds=2.0 + 4.0 * index,
                policy_inference_seconds=0.5 + index,
                graph_construction_seconds=0.25 + 0.5 * index,
            )
        rows.append(row)
    return rows


@pytest.mark.parametrize("summarize,prefix", [(_aggregate, "mean_"), (_summarize_rows, "")])
def test_reports_distinguish_mean_episode_time_from_total(summarize, prefix):
    result = summarize(_rows(timed=True))
    assert result[f"{prefix}episode_wall_seconds"] == 4.0
    assert result[f"{prefix}policy_inference_seconds"] == 1.0
    assert result[f"{prefix}graph_construction_seconds"] == 0.5
    assert result["total_episode_wall_seconds"] == 8.0


@pytest.mark.parametrize("summarize,prefix", [(_aggregate, "mean_"), (_summarize_rows, "")])
def test_reports_do_not_invent_missing_historical_timings(summarize, prefix):
    rows = _rows(timed=False)
    result = summarize(rows)
    assert f"{prefix}episode_wall_seconds" not in result
    assert "total_episode_wall_seconds" not in result
    rows[0]["episode_wall_seconds"] = 2.0
    result = summarize(rows)
    assert f"{prefix}episode_wall_seconds" not in result
    assert "total_episode_wall_seconds" not in result
