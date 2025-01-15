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

# Extract time limit
RAW_TIMELIMIT=$(scontrol show job $SLURM_JOB_ID | awk -F= '/TimeLimit/ {print $2}')

# Convert DD-HH:MM:SS to HH:MM:SS
if [[ "$RAW_TIMELIMIT" == *-* ]]; then
  # Split into days and time
  DAYS=${RAW_TIMELIMIT%-*}
  TIME=${RAW_TIMELIMIT#*-}
  # Convert days to hours
  HOURS=$((DAYS * 24 + ${TIME%%:*}))
  MINUTES=${TIME#*:}
  TIMELIMIT="$HOURS:$MINUTES"
else
  TIMELIMIT=$RAW_TIMELIMIT
fi

echo "JOB DETAILS: Running on $SLURM_NTASKS CPUs with $SRUN_CPUS_PER_TASK threads per cpu for $TIMELIMIT (hh:mm:ss)"