set -x                                                  # keep log of executed commands
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"        # assign extra environment variable to be safe 
export OPENBLAS_NUM_THREADS=1                           # avoid that OpenBLAS calls too many threads
export DATADEPS_ALWAYS_ACCEPT="true"                    # always allow data to be downloaded

# Parse command line arguments
DATA_ARG=""
for arg in "$@"; do
    if [[ $arg == --data=* ]]; then
        DATA_ARG="--data=${arg#*=}"
        break
    fi
done

MODEL_ARG=""
for arg in "$@"; do
    if [[ $arg == --model=* ]]; then
        MODEL_ARG="--model=${arg#*=}"
        break
    fi
done