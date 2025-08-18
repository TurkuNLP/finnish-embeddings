#!/bin/bash
#SBATCH --job-name=<INSERT_JOB_NAME>
#SBATCH --output=outputs/%x_%j.out
#SBATCH --account=project_<INSERT_PROJECT_ID>
#SBATCH --time=00:15:00
#SBATCH --partition=gputest
#SBATCH --gres=gpu:v100:1

# Load needed module(s)
module load pytorch

# Declare variables
export HF_HOME=<INSERT_CACHE_PATH>
MODEL=<INSERT_MODEL>
DATA_FILES=<INSERT_PATH_TO_DATA>

# Run the program
srun python3 src/embed.py $MODEL $DATA_FILES

# Shows the statistics of the run
seff $SLURM_JOBID