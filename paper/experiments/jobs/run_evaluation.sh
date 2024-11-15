#!/bin/bash
#
#SBATCH --job-name="Run Evaluation"
#SBATCH --partition=compute
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=4G
#SBATCH --account=research-eemcs-insy
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT

module load 2024r1 

source .env
source $JOB_DIR/slurm_header.sh

srun julia --project=$EXPERIMENT_DIR --threads $SLURM_CPUS_PER_TASK $EXPERIMENT_DIR/run_evaluation.jl > $LOG_DIR/run_grid.log