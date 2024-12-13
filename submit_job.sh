#!/bin/bash

#SBATCH --job-name=my_python_gpu_job     # Job name
#SBATCH --output=output.log              # Standard output file
#SBATCH --error=error.log                # Standard error file
#SBATCH --time=04:00:00                  # Maximum runtime
#SBATCH --mem=12G                        # Memory allocation
#SBATCH --gres=gpu:1                     # Request 1 GPU
#SBATCH --partition=c12m85-a100-1        # GPU partition
#SBATCH --account=csci_ga_2572-2024fa    # Account name

# Load necessary modules
module load python/3.8.5                 # Adjust to your Python version

# Run your Python script
python3 main.py