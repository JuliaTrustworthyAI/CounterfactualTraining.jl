#!/bin/bash
#
#SBATCH --job-name="Concatenate Evaluation"
#SBATCH --partition=compute
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=16G
#SBATCH --account=research-eemcs-insy
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT

module load 2024r1 

source .env
source $JOB_DIR/slurm_header.sh

srun julia --project=$EXPERIMENT_DIR --threads $SLURM_CPUS_PER_TASK $EXPERIMENT_DIR/concatenate_evaluation.jl > $LOG_DIR/concatenate_evaluation.log