#!/bin/bash
#
#SBATCH --job-name="Run Grid"
#SBATCH --partition=compute
#SBATCH --time=00:15:00
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=2G
#SBATCH --account=research-eemcs-insy
#SBATCH --mail-type=END

module load 2024r1 

source paper/experiments/jobs/slurm_header.sh

srun julia --project=$EXPERIMENT_DIR --threads $SLURM_CPUS_PER_TASK $EXPERIMENT_DIR/run_grid.jl > $LOG_DIR/run_grid.log