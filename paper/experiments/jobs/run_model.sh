#!/bin/bash
#
#SBATCH --job-name="Run Model"
#SBATCH --partition=compute
#SBATCH --time=00:10:00
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=4G
#SBATCH --account=innovation
#SBATCH --mail-type=END

module load 2024r1 

source paper/experiments/jobs/slurm_header.sh

srun julia --project=$EXPERIMENT_DIR --threads $SLURM_CPUS_PER_TASK $EXPERIMENT_DIR/run_model.jl > $LOG_DIR/run_model.log