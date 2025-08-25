#!/bin/bash
#SBATCH --job-name=paraphrase_train
#SBATCH -t 02:00:00                           # Adjust runtime as needed
#SBATCH -p grete:shared                       # Partition
#SBATCH -G A100:1                             # Request 1 A100 GPU
#SBATCH --mem-per-gpu=16G                     # More memory if needed for BART-large
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=END,FAIL                  # Notifications only on end or failure
#SBATCH --mail-user=your.name@stud.uni-goettingen.de
#SBATCH --output=./slurm_files/slurm-%x-%j.out
#SBATCH --error=./slurm_files/slurm-%x-%j.err

# Activate conda environment
source activate dnlp

# Print environment info
echo "Job running on node(s): ${SLURM_NODELIST}"
echo "Working directory: $PWD"
echo "Using Python: $(which python)"
echo "Python version: $(python --version)"
echo "CUDA devices available: $(python -c 'import torch; print(torch.cuda.device_count())')"

# Run your paraphrase fine-tuning script
python bart_generation.py --use_gpu