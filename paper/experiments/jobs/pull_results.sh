# pull_results.sh
source .env

# Initialize variables
CANDIDATE_DIR=$OUTPUT_FROM_CLUSTER_DIR
INCLUDE_PATTERNS=""

# Parse command line arguments for zsh
zparseopts -D -E -include:=include_arg

# Extract patterns from the argument if present
if (( ${#include_arg} > 0 )); then
    # Remove quotes and brackets from the argument
    INCLUDE_PATTERNS=$(echo "${include_arg[2]}" | sed 's/^=//' | tr -d "'\"[]")
    echo "Include patterns detected: $INCLUDE_PATTERNS"
fi

# Convert comma-separated patterns into rsync include patterns
RSYNC_INCLUDE=""
if [[ -n "$INCLUDE_PATTERNS" ]]; then
    # Convert comma-separated string into array
    IFS=',' read -rA PATTERN_ARRAY <<< "$INCLUDE_PATTERNS"
    
    # First include the directories
    RSYNC_INCLUDE="--include='*/' "
    
    # Then add the file patterns
    for pattern in "${PATTERN_ARRAY[@]}"; do
        RSYNC_INCLUDE="$RSYNC_INCLUDE --include='*$pattern' "
    done
    
    # Exclude everything else
    RSYNC_INCLUDE="$RSYNC_INCLUDE --exclude='*'"
    
    echo "Final rsync include/exclude patterns: $RSYNC_INCLUDE"
fi

LOCAL_STORAGE_DIR=$(bash -c "read -e -p \"Specify local storage directory (hit Enter for default $CANDIDATE_DIR): \" -r LOCAL_STORAGE_DIR && echo \"\${LOCAL_STORAGE_DIR:-$CANDIDATE_DIR}\"")
echo "Local storage directory is: $LOCAL_STORAGE_DIR"

confirm=$(bash -c "read -p \"About to copy results from long term storage directory ${LONG_TERM_STORAGE_DIR}/output to $LOCAL_STORAGE_DIR. Are you sure you want to proceed? [y/n] \" -r confirm && echo \"\$confirm\"" | tr -d '\n')

# Check the response
if [[ $confirm =~ ^[Yy](es)?$ ]]; then
    echo "Proceeding with copy..."
    if [[ -n "$RSYNC_INCLUDE" ]]; then
        RSYNC_CMD="rsync -av $RSYNC_INCLUDE paltmeyer@login.delftblue.tudelft.nl:$LONG_TERM_STORAGE_DIR/output/ $LOCAL_STORAGE_DIR/"
        echo "Debug: Executing command: $RSYNC_CMD"
        eval "$RSYNC_CMD"
    else
        rsync -av paltmeyer@login.delftblue.tudelft.nl:$LONG_TERM_STORAGE_DIR/output/ $LOCAL_STORAGE_DIR/
    fi
else
    echo "Operation cancelled"
fi