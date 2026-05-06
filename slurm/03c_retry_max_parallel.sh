#!/bin/bash
#SBATCH --account=e32706
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=03c_disvae_retry

set -euo pipefail

module load mamba/24.3.0

PROJECT_ROOT="$(cd ../ && pwd)"
cd "${PROJECT_ROOT}"

VENV_PATH="${HOME}/.venvs/image-generation-genai"
python -m venv "${VENV_PATH}"
source "${VENV_PATH}/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r "${PROJECT_ROOT}/requirements.txt"
# Pin torch/torchvision to CUDA 12.4 wheels for broader driver compatibility.
python -m pip install --upgrade --force-reinstall \
  torch torchvision \
  --index-url https://download.pytorch.org/whl/cu124

echo "===== GPU sanity check ====="
nvidia-smi || true
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available; aborting to avoid slow CPU run.")
print("device count:", torch.cuda.device_count())
print("device 0:", torch.cuda.get_device_name(0))
PY
echo "============================"

echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-1}"
echo "Running 03c_disentangled_vae.py retry"

# Keep Optuna parallelism bounded on single-GPU jobs to avoid contention.
PARALLEL_JOBS="${SLURM_CPUS_PER_TASK:-1}"
if [ "${PARALLEL_JOBS}" -gt 2 ]; then
  PARALLEL_JOBS=2
fi

python "${PROJECT_ROOT}/scripts/03c_disentangled_vae.py" \
  --tune-trials 20 \
  --tune-epochs 5 \
  --tune-subset 4000 \
  --final-epochs 60 \
  --batch-size 64 \
  --n-jobs "${PARALLEL_JOBS}" \
  --n-loader-workers "${SLURM_CPUS_PER_TASK:-1}" \
  --force-restart

# 03c already writes a combined test grid with:
# row1 content, row2 reconstruction, row3 style refs, row4 style-transfer result.
if [ ! -f "${PROJECT_ROOT}/outputs/disvae_transfer.png" ]; then
  echo "ERROR: Expected test output outputs/disvae_transfer.png not found."
  exit 1
fi
echo "Wrote test output: outputs/disvae_transfer.png (reconstruction + style transfer)"
