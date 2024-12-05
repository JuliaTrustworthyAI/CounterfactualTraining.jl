#!/bin/bash
#
#SBATCH --job-name="Run Grid"
#SBATCH --partition=compute
#SBATCH --time=01:00:00
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=2G
#SBATCH --account=research-eemcs-insy
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT

module load 2024r1 

source .env
source $JOB_DIR/slurm_header.sh

srun julia --project=$EXPERIMENT_DIR --threads $SLURM_CPUS_PER_TASK  $EXPERIMENT_DIR/run_grid.jl --config=$EXPERIMENT_DIR/run_grid_config.toml > $LOG_DIR/run_grid.log