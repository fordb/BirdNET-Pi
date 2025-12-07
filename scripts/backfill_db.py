#!/usr/bin/env python3
"""Backfill database from existing detection files."""

import os
import re
import sqlite3
from pathlib import Path

DB_PATH = "/home/ford/BirdNET-Pi/scripts/birds.db"
EXTRACTED_PATH = "/home/ford/BirdSongs/Extracted/By_Date"

def parse_filename(filename):
    """Parse detection filename to extract metadata.
    Format: Species_Name-Confidence-Date-birdnet-Time.mp3
    Example: Tufted_Titmouse-70-2025-12-07-birdnet-08:01:05.mp3
    """
    # Remove .mp3 extension
    name = filename.replace('.mp3', '')

    # Pattern: Species-Confidence-Date-birdnet-Time
    pattern = r'^(.+?)-(\d+)-(\d{4}-\d{2}-\d{2})-birdnet-(.+)$'
    match = re.match(pattern, name)

    if not match:
        return None

    species_name = match.group(1).replace('_', ' ')
    confidence = int(match.group(2)) / 100.0  # Convert percentage to decimal
    date = match.group(3)
    time = match.group(4)

    return {
        'com_name': species_name,
        'confidence': confidence,
        'date': date,
        'time': time,
        'filename': filename
    }

def get_existing_files(db_path):
    """Get set of filenames already in database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT File_Name FROM detections")
    existing = {row[0] for row in cursor.fetchall()}
    conn.close()
    return existing

def backfill_database(start_date='2025-12-07'):
    """Backfill database from detection files."""

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get files already in database
    existing_files = get_existing_files(DB_PATH)

    added = 0
    skipped = 0

    # Walk through date directories
    for date_dir in sorted(Path(EXTRACTED_PATH).glob('*')):
        if not date_dir.is_dir():
            continue

        date_str = date_dir.name
        if date_str < start_date:
            continue

        print(f"Processing {date_str}...")

        # Walk through species directories
        for species_dir in date_dir.iterdir():
            if not species_dir.is_dir():
                continue

            # Process each .mp3 file
            for mp3_file in species_dir.glob('*.mp3'):
                filename = mp3_file.name

                # Skip if already in database
                if filename in existing_files:
                    skipped += 1
                    continue

                # Parse filename
                data = parse_filename(filename)
                if not data:
                    print(f"  Skipping unparseable: {filename}")
                    continue

                # Insert into database
                try:
                    cursor.execute("""
                        INSERT INTO detections
                        (Date, Time, Com_Name, Sci_Name, Confidence, File_Name,
                         Lat, Lon, Cutoff, Week, Sens, Overlap)
                        VALUES (?, ?, ?, ?, ?, ?, 0, 0, 0, 0, 0, 0)
                    """, (
                        data['date'],
                        data['time'],
                        data['com_name'],
                        '',  # Sci_Name not available from filename
                        data['confidence'],
                        data['filename']
                    ))
                    added += 1

                    if added % 100 == 0:
                        print(f"  Added {added} detections...")
                        conn.commit()

                except sqlite3.IntegrityError as e:
                    print(f"  Error inserting {filename}: {e}")
                    continue

    conn.commit()
    conn.close()

    print(f"\nBackfill complete!")
    print(f"Added: {added} detections")
    print(f"Skipped (already in DB): {skipped}")

if __name__ == '__main__':
    print("Starting database backfill from detection files...")
    backfill_database()
