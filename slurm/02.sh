#!/bin/bash
#SBATCH --account=e32706
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --job-name=02_data_preprocess

set -euo pipefail

module load mamba/24.3.0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

VENV_PATH="${HOME}/.venvs/image-generation-genai"
python -m venv "${VENV_PATH}"
source "${VENV_PATH}/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r "${PROJECT_ROOT}/requirements.txt"

python --version
python "${PROJECT_ROOT}/scripts/02_data_preprocessing.py" \
  --image-size 128 \
  --val-frac 0.10 \
  --test-frac 0.10 \
  --workers 4