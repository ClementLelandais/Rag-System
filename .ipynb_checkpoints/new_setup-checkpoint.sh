#!/bin/bash
#SBATCH --job-name=RAG-GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

mkdir -p logs data/{raw,chunks,index}
cd ${SLURM_SUBMIT_DIR}

source /info/etu/m1/s2200573/miniconda3/etc/profile.d/conda.sh
conda activate Acollab

nvidia-smi > logs/gpu-start.log
python reindex_dataset.py --dataset bsard 2>&1 | tee logs/reindex-gpu.log
nvidia-smi >> logs/gpu-end.log
