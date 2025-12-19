#!/bin/bash
#SBATCH -J "gpu_train_job"
#SBATCH -A c00004               # Correct project name
#SBATCH -p a100x4q              # Partition name
#SBATCH -N 1                    # Number of nodes (1 node)
#SBATCH -n  64                   # Number of cores (64 cores)
#SBATCH --gres=gpu:4            # Request 4 GPUs
#SBATCH --mem=100G              # Memory per node (100 GB)
#SBATCH --time=10-00:00:00      # Maximum runtime (10 days)
#SBATCH -o slurm.%j.out         # Output file name with Job ID
#SBATCH -e slurm.%j.err         # Error file name with Job ID
#SBATCH --mail-type=END,FAIL    # Email notifications for job completion or failure
#SBATCH --mail-user=xuru0927@gmail.com  # Email address for notifications

# Load Anaconda and source the conda.sh script
module load ANACONDA/Anaconda3-2024.02-1-python-3.11
source /ari/progs/ANACONDA/Anaconda3-2024.02-1-python-3.11/etc/profile.d/conda.sh

# Activate your newly created environment
conda activate ruruCMR

# Run the Python script from the current directory
cd /ari/users/ioksuz/ruru/CMRxRecon2025/CMR2025_training/
torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr="localhost" --master_port=29500 train_distribution.py

# Optionally copy output files back to the home directory
#cp -r /ari/users/rxu/training_task2/output/distribution /ari/users/rxu

