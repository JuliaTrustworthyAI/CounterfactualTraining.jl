source .env

LOCAL_STORAGE_DIR=$(bash -c 'read -e -p "Specify local storage directory: " -r LOCAL_STORAGE_DIR && echo "$LOCAL_STORAGE_DIR"')
echo "Local storage directory is: $LOCAL_STORAGE_DIR"

confirm=$(bash -c 'read -p "About to copy results from long term storage directory $LONG_TERM_STORAGE_DIR to $LOCAL_STORAGE_DIR. Are you sure you want to proceed? [y/n] " -r confirm && echo "$confirm"')

# Check the response
if [[ $confirm =~ ^[Yy](es)?$ ]]; then
    echo "Proceeding with copy..."
    scp -pr paltmeyer@login.delftblue.tudelft.nl:$LONG_TERM_STORAGE_DIR/output $LOCAL_STORAGE_DIR
else
    echo "Operation cancelled"
fi