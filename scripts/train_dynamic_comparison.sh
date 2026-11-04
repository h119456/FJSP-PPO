#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
training_python="${PYTHON:-/root/miniconda3/envs/datascience/bin/python}"
training_mode="${1:-check}"
if [[ $# -gt 1 ]]; then
  echo "Usage: bash scripts/train_dynamic_comparison.sh [check|mlp|gnn|both]" >&2
  exit 2
fi
case "$training_mode" in
  check|mlp|gnn|both) ;;
  *) echo "Usage: bash scripts/train_dynamic_comparison.sh [check|mlp|gnn|both]" >&2; exit 2 ;;
esac

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"

"$training_python" scripts/validate_dynamic_dataset.py
"$training_python" scripts/check_comparison.py \
  --mlp-config configs/dataset_compare_mlp_dynamic.json \
  --gnn-config configs/dataset_compare_gnn_dynamic.json \
  --output outputs/dynamic_training/preflight.json > /dev/null
"$training_python" -c 'import torch; n = torch.cuda.device_count(); assert torch.cuda.is_available() and n == 8, f"Expected 8 visible CUDA GPUs, found {n}"; print("Dynamic preflight passed: checkpoints compatible; 8 CUDA GPUs visible.")'
if [[ "$training_mode" == check ]]; then
  exit 0
fi

training_stamp="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
training_batch="$(mktemp -d "$project_root/outputs/dynamic_training/launch_${training_stamp}_XXXXXX")"
cp outputs/dynamic_training/preflight.json "$training_batch/preflight.json"
comparison_models=("$training_mode")
if [[ "$training_mode" == both ]]; then
  comparison_models=(mlp gnn)
fi
echo "Launch records: $training_batch"
for comparison_model in "${comparison_models[@]}"; do
  cp "configs/dataset_compare_${comparison_model}_dynamic.json" "$training_batch/${comparison_model}_config.json"
done
for comparison_model in "${comparison_models[@]}"; do
  "$training_python" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
    -m ppo_rs.train --config "$training_batch/${comparison_model}_config.json" \
    2>&1 | tee "$training_batch/${comparison_model}.log"
done
