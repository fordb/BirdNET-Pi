#!/bin/bash

# Define user and host
USER="ford"
HOST="birdnetpi.local"

# Define the base remote path
REMOTE_PATH="/home/ford"

# Define the local base path for the backup
TODAY=$(date +%Y-%m-%d)
LOCAL_PATH="/Users/ford/Desktop/backup-$TODAY"

# Remote commands to zip directories with progress
ssh ${USER}@${HOST} "tar -cf - -C ${REMOTE_PATH} BirdSongs/Extracted/By_Date BirdSongs/Extracted/Charts | pv -s $(du -sb ${REMOTE_PATH}/BirdSongs/Extracted/By_Date ${REMOTE_PATH}/BirdSongs/Extracted/Charts | awk '{s+=$1} END {print s}') | gzip > /tmp/birdnet_backup.tar.gz"

# Create local backup directory
mkdir -p "$LOCAL_PATH"

# List of individual files to backup
FILES_TO_BACKUP=(
    "$REMOTE_PATH/BirdNET-Pi/birdnet.conf"
    "$REMOTE_PATH/BirdNET-Pi/apprise.txt"
    "$REMOTE_PATH/BirdNET-Pi/scripts/birds.db"
)

# List of directories to backup
DIRECTORIES_TO_BACKUP=(
    "$REMOTE_PATH/BirdSongs/Extracted/By_Date"
    "$REMOTE_PATH/BirdSongs/Extracted/Charts"
)

# Copy files using scp
for FILE in "${FILES_TO_BACKUP[@]}"; do
    scp "${USER}@${HOST}:${FILE}" "$LOCAL_PATH/"
done

# Transfer the compressed file
scp "${USER}@${HOST}:/tmp/birdnet_backup.tar.gz" "$LOCAL_PATH/"

echo "Backup completed successfully."