#!/usr/bin/bash
#SBATCH --job-name tokenization
#SBATCH --account OPEN-29-45
#SBATCH --partition qgpu
#SBATCH --gpus 1
#SBATCH --time=1-00:00:00 # 1 days, 0 hours, 0 minutes, 0 seconds 

ml purge
ml load Python/3.11.5-GCCcore-13.2.0
. ./venv/bin/activate

cd dataset-playground/azure_data
srun python3 merge_all_to_one_bin.py

