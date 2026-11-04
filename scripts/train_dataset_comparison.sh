#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

python scripts/check_comparison.py \
  --mlp-config configs/dataset_compare_mlp.json \
  --gnn-config configs/dataset_compare_gnn.json > /dev/null

for comparison_model in mlp gnn; do
  python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
    -m ppo_rs.train --config "configs/dataset_compare_${comparison_model}.json"
done
