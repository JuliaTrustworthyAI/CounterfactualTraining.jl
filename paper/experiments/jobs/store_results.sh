source .env

# Copy results to long-term storage:
if [ $PWD = $CLUSTER_WORK_DIR ] ; then
    echo 'Copying results to long term storage directory: $LONG_TERM_STORAGE_DIR'
    cp -rf -n $OUTPUT_DIR $LONG_TERM_STORAGE_DIR
else
    echo 'It seems you are not in the cluster work directory. Skipping copy to long term storage.'
fi