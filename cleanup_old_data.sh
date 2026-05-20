#!/bin/bash

BASE_DIR="/home/melosavrao/insect_ui/insect_tracker/static"

echo "🧹 Cleanup started at $(date)"

# Delete folders older than yesterday 11 PM
find "$BASE_DIR/output" -mindepth 1 -maxdepth 1 -type d ! -newermt "yesterday 23:00" -exec rm -rf {} +

find "$BASE_DIR/trap_images" -mindepth 1 -maxdepth 1 -type d ! -newermt "yesterday 23:00" -exec rm -rf {} +

echo " Cleanup finished at $(date)"
