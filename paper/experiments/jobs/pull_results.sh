#!/bin/zsh

source .env

# Default values
CANDIDATE_DIR=$OUTPUT_FROM_CLUSTER_DIR
FILETYPES=()

# Parse command line arguments
# zsh requires different array handling
zparseopts -D -E -filetype:=opt_filetype

if [[ -n "$opt_filetype" ]]; then
    # Extract the value after the = sign
    types=${opt_filetype[2]}
    # Remove brackets and quotes
    types=${types//[\[\]\"\']/}
    # Split by comma into array
    FILETYPES=(${(s:,:)types})
fi

# Get local storage directory
LOCAL_STORAGE_DIR=$(zsh -c "read \"?Specify local storage directory (hit Enter for default $CANDIDATE_DIR): \" LOCAL_STORAGE_DIR; echo \${LOCAL_STORAGE_DIR:-$CANDIDATE_DIR}")
echo "Local storage directory is: $LOCAL_STORAGE_DIR"

# Build include pattern for rsync if filetypes are specified
RSYNC_INCLUDE=""
if (( ${#FILETYPES} > 0 )); then
    echo "Will only copy files with extensions: ${(j:, :)FILETYPES}"
    # Create include patterns for each file type
    for type in $FILETYPES; do
        RSYNC_INCLUDE="$RSYNC_INCLUDE --include='*.$type'"
    done
    # Exclude everything else
    RSYNC_INCLUDE="$RSYNC_INCLUDE --exclude='*'"
fi

# Show confirmation prompt
read "confirm?About to copy results from long term storage directory ${OUTPUT_DIR} to $LOCAL_STORAGE_DIR. Are you sure you want to proceed? [y/n] "

# Check the response
if [[ $confirm =~ ^[Yy](es)?$ ]]; then
    echo "Proceeding with copy..."
    if [[ -n "$RSYNC_INCLUDE" ]]; then
        # Use eval to properly handle the include/exclude patterns
        eval "rsync -av $RSYNC_INCLUDE paltmeyer@login.delftblue.tudelft.nl:$CLUSTER_WORK_DIR/$OUTPUT_DIR/ $LOCAL_STORAGE_DIR/"
    else
        # Original behavior when no filetypes specified
        rsync -av paltmeyer@login.delftblue.tudelft.nl:$CLUSTER_WORK_DIR/$OUTPUT_DIR/ $LOCAL_STORAGE_DIR/
    fi
else
    echo "Operation cancelled"
fi