#!/bin/bash
#
#SBATCH --job-name="Test Grid"
#SBATCH --partition=compute
#SBATCH --time=00:10:00
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --account=innovation
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT

module load 2024r1 

source .env
source $JOB_DIR/slurm_header.sh

srun julia --project=$EXPERIMENT_DIR --threads $SLURM_CPUS_PER_TASK $EXPERIMENT_DIR/run_grid.jl --config=$CONFIG > $LOG_DIR/run_grid.log