source .env

CANDIDATE_DIR=paper/experiments/output/
LOCAL_STORAGE_DIR=$(bash -c "read -e -p \"Specify local storage directory (hit Enter for default $CANDIDATE_DIR): \" -r LOCAL_STORAGE_DIR && echo \"\${LOCAL_STORAGE_DIR:-$CANDIDATE_DIR}\"")
echo "Local storage directory is: $LOCAL_STORAGE_DIR"

confirm=$(bash -c "read -p \"About to copy results from long term storage directory ${LONG_TERM_STORAGE_DIR:-unknown} to $LOCAL_STORAGE_DIR. Are you sure you want to proceed? [y/n] \" -r confirm && echo \"\$confirm\"" | tr -d '\n')

# Check the response
if [[ $confirm =~ ^[Yy](es)?$ ]]; then
    echo "Proceeding with copy..."
    rsync -av paltmeyer@login.delftblue.tudelft.nl:$LONG_TERM_STORAGE_DIR/output/ $LOCAL_STORAGE_DIR/
else
    echo "Operation cancelled"
fi