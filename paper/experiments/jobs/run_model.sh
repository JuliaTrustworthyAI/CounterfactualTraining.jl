#!/bin/bash
#
#SBATCH --job-name="Run Model"
#SBATCH --partition=compute
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=2G
#SBATCH --account=innovation
#SBATCH --mail-type=END

module load 2024r1 

source paper/experiments/jobs/slurm_header.sh

srun julia --project=$EXPERIMENT_DIR --threads $SLURM_CPUS_PER_TASK $EXPERIMENT_DIR/run_model.jl > $LOG_DIR/run_model.log