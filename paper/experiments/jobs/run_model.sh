#!/bin/bash
#
#SBATCH --job-name="Run Model"
#SBATCH --partition=compute
#SBATCH --time=00:10:00
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=2G
#SBATCH --account=innovation
#SBATCH --mail-type=END

module load 2024r1 

source experiments/slurm_header.sh

srun julia --project=$EXPERIMENT_FOLDER --threads $SLURM_CPUS_PER_TASK $EXPERIMENT_FOLDER/run_model.jl > $EXPERIMENT_FOLDER/logs/run_model.log