import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from scripts import wait_train_gnn_gate_set1 as waiter


def source(tmp_path, status="complete"):
    path = tmp_path / "outputs/simplestate_jingtai_set1/pipeline_status.json"
    path.parent.mkdir(parents=True)
    data = dict(status=status, active_job="dynamic/GNN",
                jobs={job: {"status": "complete"} for job in waiter.JOBS})
    path.write_text(json.dumps(data))
    return path


def test_stage_gap_does_not_start_gate(tmp_path, monkeypatch):
    source(tmp_path, "training")
    probe = Mock(return_value=[])
    monkeypatch.setattr(waiter, "gpu_processes", probe)
    assert waiter.readiness(tmp_path)[0] is False
    probe.assert_not_called()


def test_all_jobs_complete_and_gpus_idle_required(tmp_path, monkeypatch):
    path = source(tmp_path)
    monkeypatch.setattr(waiter, "gpu_processes", lambda: ["uuid,pid"])
    assert waiter.readiness(tmp_path)[0] is False
    monkeypatch.setattr(waiter, "gpu_processes", lambda: [])
    assert waiter.readiness(tmp_path)[0] is True
    data = json.loads(path.read_text())
    data['jobs']['dynamic/GNN']['status'] = 'training'
    path.write_text(json.dumps(data))
    with pytest.raises(RuntimeError, match='four jobs'):
        waiter.readiness(tmp_path)


@pytest.mark.parametrize("status", ["failed", "interrupted"])
def test_failed_previous_chain_does_not_launch(tmp_path, status):
    source(tmp_path, status)
    with pytest.raises(RuntimeError, match='stopped'):
        waiter.readiness(tmp_path)


def test_existing_target_refused(tmp_path):
    (tmp_path / "outputs/GNN+gate_set1").mkdir(parents=True)
    with pytest.raises(RuntimeError, match='already exists'):
        waiter.readiness(tmp_path)


def test_gpu_uuid_mapping_ignores_other_cards(monkeypatch):
    outputs = iter(["0, GPU-zero\n1, GPU-one\n2, GPU-two\n", "GPU-two, 22\nGPU-one, 11\n"])
    monkeypatch.setattr(waiter.subprocess, 'check_output', lambda *a, **k: next(outputs))
    assert waiter.gpu_processes() == ['GPU-one, 11']


@pytest.mark.parametrize('returncode', [0, 1])
def test_waits_two_probes_then_runs_chain_once(tmp_path, monkeypatch, returncode):
    probes = iter([(False, 'previous training'), (True, 'idle'), (True, 'idle')])
    monkeypatch.setattr(waiter, 'readiness', lambda root: next(probes))
    sleep = Mock()
    monkeypatch.setattr(waiter.time, 'sleep', sleep)
    monkeypatch.setattr(waiter.signal, 'signal', Mock())
    child = Mock(pid=123)
    child.wait.return_value = returncode
    launch = Mock(return_value=child)
    monkeypatch.setattr(waiter.subprocess, 'Popen', launch)
    assert waiter.run(tmp_path, 60) == returncode
    launch.assert_called_once_with(['bash', str(tmp_path/'scripts/train_gnn_gate_set1.sh')], cwd=tmp_path, start_new_session=True)
    assert sleep.call_count == 2
    state = json.loads((tmp_path/'outputs/gnn_gate_set1_waiter_status.json').read_text())
    assert state['status'] == ('complete' if returncode == 0 else 'failed')


def test_duplicate_waiter_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(waiter.fcntl, 'flock', Mock(side_effect=BlockingIOError))
    monkeypatch.setattr(waiter.subprocess, 'Popen', Mock(side_effect=AssertionError('must not launch')))
    assert waiter.run(tmp_path, 60) == 1
