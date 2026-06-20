#!/bin/bash
#SBATCH -N 2                          # Number of nodes
#SBATCH -c 12                         # Number of CPU cores
#SBATCH --mem=16gb                   # RAM memory
#SBATCH --time=3:00:00               # Maximum runtime (3 hours)
#SBATCH --job-name=gamma             # Job name
#SBATCH --partition=lem-gpu          # Partition with GPU
#SBATCH --gres=gpu:hopper:4          # Request 4 Hopper GPUs

# Activate virtual environment
echo "Script started at $(date)"
source /usr/local/sbin/modules.sh
ml Python/3.11.5-GCCcore-13.2.0
echo "Modules loaded at $(date)"

# Install required Python packages
pip install torch torchvision scikit-learn pandas nibabel openpyxl monai
pip install numpy pandas nibabel torch scikit-learn opencv-python

# Run Python script
echo "Starting Python script at $(date)"
python3 /home/kacjan7732/workspace/train_model.py
echo "Python script finished at $(date)"
