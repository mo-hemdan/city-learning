#!/bin/bash

CITY_GRIDS_JSON="./city_grids.json"
LOG_DIR="./logs"

mkdir -p "$LOG_DIR"

areas=$(python -c "import json; print('\n'.join(json.load(open('$CITY_GRIDS_JSON')).keys()))")
areas_array=($areas)
total=${#areas_array[@]}

echo "Found $total areas"

for area in "${areas_array[@]}"; do
    echo "Splitting: $area"

    python 1.2_split_train_test.py --city "$area" --setting transductive \
        > "$LOG_DIR/${area}_split.log" 2>&1

    if [ $? -eq 0 ]; then
        echo "✓ $area completed successfully"
    else
        echo "✗ $area failed — check $LOG_DIR/${area}_split.log"
    fi
done

echo "=========================================="
echo "All areas done."
