#!/bin/bash

#SBATCH --job-name=ensdV_b
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -c 2
#SBATCH --mem=4gb
#SBATCH --time=1:00:00

# Activate your virtual environment
source /usr/local/sbin/modules.sh
ml Python/3.11.5-GCCcore-13.2.0

# Install dependencies (if not already installed in your venv)
pip install torch torchvision scikit-learn pandas nibabel openpyxl scikit-learn

# Run your Python script
python3 /home/kacjan7732/workspace/model_train.py