#!/bin/bash
#
#SBATCH --job-name="Run Grid"
#SBATCH --partition=compute
#SBATCH --time=00:30:00
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --account=research-eemcs-insy
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

module load 2024r1 

source paper/experiments/jobs/slurm_header.sh

srun julia --project=$EXPERIMENT_DIR --threads $SLURM_CPUS_PER_TASK $EXPERIMENT_DIR/run_grid.jl > $LOG_DIR/run_grid.log

# Only resubmit if the job timed out
if grep -q "DUE TO TIME LIMIT" slurm-${SLURM_JOB_ID}.out 2>/dev/null; then
    echo "Job timed out. Resubmitting..."
    sbatch $0
fi