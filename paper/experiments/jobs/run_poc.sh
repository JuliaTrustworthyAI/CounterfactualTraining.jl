#!/bin/bash
#
#SBATCH --job-name="Run POC"
#SBATCH --partition=compute
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=2G
#SBATCH --account=research-eemcs-insy
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT

module load 2024r1 

source .env
source $JOB_DIR/slurm_header.sh

srun julia --project=$EXPERIMENT_DIR --threads $SLURM_CPUS_PER_TASK $EXPERIMENT_DIR/run_poc.jl > $LOG_DIR/run_poc.log