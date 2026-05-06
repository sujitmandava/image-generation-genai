#!/bin/bash
#SBATCH --account=e32706  ## Required: your Slurm account name, i.e. eXXXX, pXXXX or bXXXX
#SBATCH --partition=gengpu ## Required: buyin, short, normal, long, gengpu, genhimem, etc.
#SBATCH --gres=gpu:1  
#SBATCH --time=8:00:00       ## Full-dataset download can take substantially longer
#SBATCH --nodes=1             ## How many computers/nodes do you need? Usually 1
#SBATCH --ntasks=1            ## How many CPUs or processors do you need? (default value 1)
#SBATCH --mem=16G             ## More headroom for full dataset ingestion
#SBATCH --job-name=data_download       ## Used to identify the job 

set -euo pipefail

module load mamba/24.3.0

PROJECT_ROOT="$(cd ../ && pwd)"
cd "${PROJECT_ROOT}"

VENV_PATH="${HOME}/.venvs/image-generation-genai"
python -m venv "${VENV_PATH}"
source "${VENV_PATH}/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r "${PROJECT_ROOT}/requirements.txt"

python --version

# 01_data_acquisition.py now downloads the full dataset by default.
# Configure behavior by editing constants inside the script.
python "${PROJECT_ROOT}/scripts/01_data_acquisition.py"
