#!/usr/bin/env bash
# Manual full-state dataset1 cost0 GNN+gate: static, then dynamic, GPUs 0-1.
set -euo pipefail

gate_project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$gate_project_root"
gate_python=/root/miniconda3/envs/datascience/bin/python
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH="$gate_project_root/src:$gate_project_root${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1

"$gate_python" - <<'PY'
import json
from pathlib import Path

root = Path.cwd()
previous = root / 'outputs/simplestate_jingtai_set1/pipeline_status.json'
if previous.exists() and json.loads(previous.read_text())['status'] != 'complete':
    raise SystemExit('GPU0-1 dataset1 chain has not completed; retry after it finishes.')
output = root / 'outputs/GNN+gate_set1'
if output.exists():
    raise SystemExit(f'Refusing to overwrite existing run: {output}')
output.mkdir(exist_ok=False)
for phase in ('static', 'dynamic'):
    config = json.loads((root / f'configs/physical_gate/full_gnn_{phase}.json').read_text())
    directory = output / phase
    directory.mkdir()
    config.update(distributed_world_size=2, output_dir=str(directory), output_dir_mode='existing')
    for key in ('train_instances', 'eval_instances', 'test_instances'):
        config[key] = str(root / config[key])
    if phase == 'dynamic':
        config['initialize_from'] = str(output / 'static/checkpoints/best.pt')
    (directory / 'training_config.json').write_text(json.dumps(config, indent=2) + '\n')
print(f'Configurations saved: {output}', flush=True)
PY

for gate_phase in static dynamic; do
  "$gate_python" - "$gate_phase" <<'PY'
import json
from pathlib import Path
import sys
from scripts.train_state_comparison import audit_run_configs

output = Path('outputs/GNN+gate_set1')
phase = sys.argv[1]
if phase == 'dynamic':
    static = output / 'static'
    config = json.loads((static / 'training_config.json').read_text())
    history = json.loads((static / 'training_history.json').read_text())
    if len(history) != config['episodes'] or int(history[-1]['episode']) != config['episodes']:
        raise SystemExit('Static history incomplete; dynamic training will not start.')
    for name in ('best', 'latest', 'final'):
        if not (static / f'checkpoints/{name}.pt').is_file():
            raise SystemExit(f'Static checkpoint missing: {name}')
config = json.loads((output / phase / 'training_config.json').read_text())
report = audit_run_configs({'gnn': config})
(output / phase / 'preflight.json').write_text(json.dumps(report, indent=2) + '\n')
if not report['passed']:
    raise SystemExit(str(report['errors']))
print(f'{phase}: preflight passed', flush=True)
PY
  "$gate_python" -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node=2 \
    -m ppo_rs.train \
    --config "outputs/GNN+gate_set1/$gate_phase/training_config.json" \
    2>&1 | tee "outputs/GNN+gate_set1/$gate_phase/training.log"
done

printf '%s\n' 'GNN+gate dataset1 static and dynamic training completed.'
