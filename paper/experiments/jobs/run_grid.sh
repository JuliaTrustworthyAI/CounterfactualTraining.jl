#!/bin/bash
#
#SBATCH --job-name="Run Grid"
#SBATCH --partition=compute
#SBATCH --time=00:30:00
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --account=research-eemcs-insy
#SBATCH --mail-type=END,REQUEUE
#SBATCH --requeue

module load 2024r1 

source paper/experiments/jobs/slurm_header.sh

# Only allow requeue on timeout, not other failures
# This needs to run before the main job
if [[ ! -z "${SLURM_JOB_ID}" && "${SLURM_RESTART_COUNT}" != "" ]]; then
    # Check if the previous attempt failed due to timeout
    if ! grep -q "DUE TO TIME LIMIT" $LOG_DIR/slurm-${SLURM_JOB_ID}.out 2>/dev/null; then
        # If it wasn't a timeout, prevent requeuing
        scontrol requeue unrequeable ${SLURM_JOB_ID}
    fi
fi

srun julia --project=$EXPERIMENT_DIR --threads $SLURM_CPUS_PER_TASK $EXPERIMENT_DIR/run_grid.jl > $LOG_DIR/run_grid.log