#!/bin/bash
#SBATCH --account=e32706  ## Required: your Slurm account name, i.e. eXXXX, pXXXX or bXXXX
#SBATCH --partition=gengpu ## Required: buyin, short, normal, long, gengpu, genhimem, etc.
#SBATCH --gres=gpu:1  
#SBATCH --time=0:10:00       ## Required: How long will the job need to run?  Limits vary by partition
#SBATCH --nodes=1             ## How many computers/nodes do you need? Usually 1
#SBATCH --ntasks=1            ## How many CPUs or processors do you need? (default value 1)
#SBATCH --mem=2G              ## How much RAM do you need per computer/node? G = gigabytes
#SBATCH --job-name=test_quest_run       ## Used to identify the job 

module load mamba/24.3.0
python --version
python -c "print('hello')" > output.txt
