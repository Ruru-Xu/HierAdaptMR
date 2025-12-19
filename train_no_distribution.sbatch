#!/bin/bash
#SBATCH -J "gpu train job"     
#SBATCH -A c00002               # Correct project name
#SBATCH -p a100q                # Partition name
#SBATCH -N 3                    # Number of nodes
#SBATCH -n 192                   # Number of cores
#SBATCH --gres=gpu:1            # Request 1 GPU
#SBATCH --mem=100G              # Memory per node
#SBATCH --time=10-00:00:00      # Maximum runtime (10 days)
#SBATCH -o slurm.%j.out         # Output file name with Job ID
#SBATCH -e slurm.%j.err         # Error file name with Job ID
#SBATCH --mail-type=END,FAIL    # Email notifications for job completion or failure
#SBATCH --mail-user=xuru0927@gmail.com  # Email address for notifications

# Load Anaconda and source the conda.sh script
module load ANACONDA/Anaconda3-2023.09-0-python-3.11
source /ari/progs/ANACONDA/Anaconda3-2023.09-0-python-3.11/etc/profile.d/conda.sh

# Activate your newly created environment
conda activate ~/ruruCMR

# Run the Python script from the current directory
cd /ari/users/rxu/training_task2
python train_no_distribution.py --data_path /ari/users/rxu/datasets/MultiCoil

# Optionally copy output files back to the home directory
cp -r /ari/users/rxu/training_task2/output /ari/users/rxu

