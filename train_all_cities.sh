#!/bin/bash

CITIES_JSON="./cities.json"
LOG_DIR="./logs"

mkdir -p "$LOG_DIR"

cities=$(python -c "import json; print('\n'.join(json.load(open('$CITIES_JSON')).keys()))")

for city in $cities; do
    echo "=========================================="
    echo "Training city: $city"
    echo "=========================================="

    python 2_train_on_graphs.py --city "$city" 2>&1 | tee "$LOG_DIR/${city}.log"

    if [$? -eq 0]; then 
        echo "✓ $city completed successfully"
    else
        echo "✗ $city failed — check $LOG_DIR/${city}.log"
    fi
done

echo "All cities done."