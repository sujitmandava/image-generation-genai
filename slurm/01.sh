#!/bin/bash
#SBATCH --account=e32706  ## Required: your Slurm account name, i.e. eXXXX, pXXXX or bXXXX
#SBATCH --partition=gengpu ## Required: buyin, short, normal, long, gengpu, genhimem, etc.
#SBATCH --gres=gpu:1  
#SBATCH --time=0:10:00       ## Required: How long will the job need to run?  Limits vary by partition
#SBATCH --nodes=1             ## How many computers/nodes do you need? Usually 1
#SBATCH --ntasks=1            ## How many CPUs or processors do you need? (default value 1)
#SBATCH --mem=2G              ## How much RAM do you need per computer/node? G = gigabytes
#SBATCH --job-name=test_quest_run       ## Used to identify the job 

set -euo pipefail

module load mamba/24.3.0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

VENV_PATH="${HOME}/.venvs/image-generation-genai"
python -m venv "${VENV_PATH}"
source "${VENV_PATH}/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python --version
python "${PROJECT_ROOT}/scripts/01_data_acquisition.py" --fraction 0.10 --n-styles 8 --max-side-px 512 --force False --seed 42