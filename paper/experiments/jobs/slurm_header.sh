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

# Convert SLURM_TIMELIMIT (in seconds) to HH:MM:SS
if [[ -n "$SLURM_TIMELIMIT" ]]; then
  HOURS=$((SLURM_TIMELIMIT / 3600))
  MINUTES=$(( (SLURM_TIMELIMIT % 3600) / 60 ))
  SECONDS=$((SLURM_TIMELIMIT % 60))
  TIMELIMIT=$(printf "%02d:%02d:%02d" $HOURS $MINUTES $SECONDS)
else
  TIMELIMIT="Unknown"
fi

echo "JOB DETAILS: Running on $SLURM_NTASKS CPUs with $SRUN_CPUS_PER_TASK threads per cpu for $TIMELIMIT (hh:mm:ss)"