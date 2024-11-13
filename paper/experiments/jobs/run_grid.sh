#!/bin/bash
#
#SBATCH --job-name="Run Grid"
#SBATCH --partition=compute
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --account=research-eemcs-insy
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT

module load 2024r1 

source paper/experiments/jobs/slurm_header.sh
source .env

srun julia --project=$EXPERIMENT_DIR --threads $SLURM_CPUS_PER_TASK $EXPERIMENT_DIR/run_grid.jl > $LOG_DIR/run_grid.log

# Copy results to long-term storage:
if [ $PWD = $CLUSTER_WORK_DIR ] ; then
    echo 'Copying results to long term storage directory: $LONG_TERM_STORAGE_DIR'
    cp -rf -n $OUTPUT_DIR $LONG_TERM_STORAGE_DIR
fi