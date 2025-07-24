# pull_results.sh
source .env

# Initialize variables
CANDIDATE_DIR=$OUTPUT_FROM_CLUSTER_DIR
INCLUDE_PATTERNS=""

# Parse command line arguments for zsh
zparseopts -D -E -include:=include_arg -exclude:=exclude_arg

# Extract patterns from the argument if present
if (( ${#include_arg} > 0 )); then
    # Remove quotes and brackets from the argument
    INCLUDE_PATTERNS=$(echo "${include_arg[2]}" | sed 's/^=//' | tr -d "'\"[]")
    echo "Include patterns detected: $INCLUDE_PATTERNS"
fi

# Extract exclude patterns if present
if (( ${#exclude_arg} > 0 )); then
    EXCLUDE_PATTERNS=$(echo "${exclude_arg[2]}" | sed 's/^=//' | tr -d "'\"[]")
    echo "Debug: Exclude patterns detected: $EXCLUDE_PATTERNS"
fi

# Build rsync patterns
RSYNC_PATTERNS="--include='*/' " # Always include directories

# Add include patterns if specified
if [[ -n "$INCLUDE_PATTERNS" ]]; then
    IFS=',' read -rA PATTERN_ARRAY <<< "$INCLUDE_PATTERNS"
    for pattern in "${PATTERN_ARRAY[@]}"; do
        RSYNC_PATTERNS="$RSYNC_PATTERNS --include='*$pattern' "
    done
    RSYNC_PATTERNS="$RSYNC_PATTERNS --exclude='*' "  # Exclude everything else if we have includes
elif [[ -n "$EXCLUDE_PATTERNS" ]]; then
    # If we only have excludes (no includes), don't add the final exclude='*'
    IFS=',' read -rA PATTERN_ARRAY <<< "$EXCLUDE_PATTERNS"
    for pattern in "${PATTERN_ARRAY[@]}"; do
        RSYNC_PATTERNS="$RSYNC_PATTERNS --exclude='*$pattern' "
    done
fi

# If we have both includes and excludes, add excludes after the include patterns
if [[ -n "$INCLUDE_PATTERNS" ]] && [[ -n "$EXCLUDE_PATTERNS" ]]; then
    IFS=',' read -rA PATTERN_ARRAY <<< "$EXCLUDE_PATTERNS"
    for pattern in "${PATTERN_ARRAY[@]}"; do
        RSYNC_PATTERNS="$RSYNC_PATTERNS --exclude='*$pattern' "
    done
fi

echo "Debug: Final rsync patterns: $RSYNC_PATTERNS"

LOCAL_STORAGE_DIR=$(bash -c "read -e -p \"Specify local storage directory (hit Enter for default $CANDIDATE_DIR): \" -r LOCAL_STORAGE_DIR && echo \"\${LOCAL_STORAGE_DIR:-$CANDIDATE_DIR}\"")
echo "Local storage directory is: $LOCAL_STORAGE_DIR"

confirm=$(bash -c "read -p \"About to copy results from long term storage directory ${LONG_TERM_STORAGE_DIR}/output to $LOCAL_STORAGE_DIR. Are you sure you want to proceed? [y/n] \" -r confirm && echo \"\$confirm\"" | tr -d '\n')

# Check the response
if [[ $confirm =~ ^[Yy](es)?$ ]]; then
    echo "Proceeding with copy..."
    if [[ -n "$RSYNC_PATTERNS" ]]; then
        # ID removed below for double-blind
        RSYNC_CMD="rsync -av $RSYNC_PATTERNS anonymous:$LONG_TERM_STORAGE_DIR/output/ $LOCAL_STORAGE_DIR/"
        echo "Debug: Executing command: $RSYNC_CMD"
        eval "$RSYNC_CMD"
    else
        # ID removed below for double-blind
        rsync -av anonymous:$LONG_TERM_STORAGE_DIR/output/ $LOCAL_STORAGE_DIR/
    fi
else
    echo "Operation cancelled"
fi
