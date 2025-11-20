#!/bin/bash
set -e

# BirdNET-Pi Local Backup Script
# Backs up database, configs, and audio to local Mac
# Keeps last 4 weekly backups to save space

# Define user and host
USER="ford"
HOST="birdnetpi.local"

# Define the base remote path
REMOTE_PATH="/home/ford"

# Define the local base path for the backup
TODAY=$(date +%Y-%m-%d)
BACKUP_ROOT="/Users/ford/BirdNET-Backups"
LOCAL_PATH="$BACKUP_ROOT/backup-$TODAY"

echo "BirdNET-Pi Backup Starting..."
echo "Date: $TODAY"
echo ""

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
echo "Transferring audio archive..."
scp "${USER}@${HOST}:/tmp/birdnet_backup.tar.gz" "$LOCAL_PATH/"

# Clean up remote temp file
ssh ${USER}@${HOST} "rm /tmp/birdnet_backup.tar.gz"

echo ""
echo "✓ Backup completed successfully to: $LOCAL_PATH"
echo ""

# Keep only the last 4 backups (monthly retention)
echo "Cleaning up old backups (keeping last 4)..."
cd "$BACKUP_ROOT"
ls -dt backup-* | tail -n +5 | xargs -r rm -rf
echo "✓ Old backups cleaned up"
echo ""

# Show backup summary
echo "Backup Summary:"
echo "==============="
echo "Latest backup: $LOCAL_PATH"
du -sh "$LOCAL_PATH"
echo ""
echo "All backups:"
ls -lht "$BACKUP_ROOT" | grep "^d"
echo ""
echo "Total backup size:"
du -sh "$BACKUP_ROOT"