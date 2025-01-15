set -x                                                  # keep log of executed commands
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"        # assign extra environment variable to be safe 
export OPENBLAS_NUM_THREADS=1                           # avoid that OpenBLAS calls too many threads
export DATADEPS_ALWAYS_ACCEPT="true"                    # always allow data to be downloaded

# Initialize empty string for additional arguments
EXTRA_ARGS=""

# Loop through all arguments
for arg in "$@"; do
    # Check if argument starts with --
    if [[ $arg == --* ]]; then
        EXTRA_ARGS="$EXTRA_ARGS $arg"
    fi
done

TIME=$(squeue -j $SLURM_JOB_ID -h --Format TimeLimit)

echo "JOB DETAILS: Running on $SLURM_NTASKS CPUs with $SRUN_CPUS_PER_TASK threads per cpu for $TIME (mm:ss)"