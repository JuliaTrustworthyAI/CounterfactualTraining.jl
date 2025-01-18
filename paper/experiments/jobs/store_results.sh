source .env

#!/bin/bash

# Function to prompt for confirmation
confirm_delete() {
    local dest_dir="$1"
    echo "WARNING: This will delete files in $dest_dir that don't exist in the source."
    echo "The following files will be deleted:"
    rsync -avn --delete --no-perms "$OUTPUT_DIR/" "$dest_dir/output/" | grep "^deleting"
    
    read -p "Do you want to proceed? (y/N) " response
    case "$response" in
        [yY][eE][sS]|[yY]) 
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

# Copy results to long-term storage:
if [ "$PWD" = "$CLUSTER_WORK_DIR" ]; then
    echo "Copying results to long term storage directory: $LONG_TERM_STORAGE_DIR"
    
    # Check if --delete flag was provided
    if [[ "$1" == "--delete" ]]; then
        if confirm_delete "$LONG_TERM_STORAGE_DIR"; then
            rsync -av --delete --no-perms "$OUTPUT_DIR/" "$LONG_TERM_STORAGE_DIR/output/"
        else
            echo "Sync cancelled. Proceeding with normal copy without deletion."
            rsync -av --no-perms "$OUTPUT_DIR/" "$LONG_TERM_STORAGE_DIR/output/"
        fi
    else
        rsync -av --no-perms "$OUTPUT_DIR/" "$LONG_TERM_STORAGE_DIR/output/"
    fi
else
    echo 'It seems you are not in the cluster work directory. Skipping copy to long term storage.'
fi