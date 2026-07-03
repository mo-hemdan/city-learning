#!/bin/bash

CITIES_JSON="./cities.json"
DATA_DIR="./data/raw_data"
CHECKPOINT_DIR="./checkpoints"
OUTPUT_DIR="./data/imputed_data"
RESULTS_DIR="./results"
LOG_DIR="./logs/infer"
DEVICE="cpu"

mkdir -p "$OUTPUT_DIR" "$RESULTS_DIR" "$LOG_DIR"

cities=$(python3 -c "import json; print('\n'.join(json.load(open('$CITIES_JSON')).keys()))")

for source_city in $cities; do
    for target_city in $cities; do
        echo "=========================================="
        echo "  source=$source_city  →  target=$target_city"
        echo "=========================================="

        python 3_predict_on_graphs.py \
            --source_city "$source_city" \
            --target_city "$target_city" \
            --data_dir "$DATA_DIR" \
            --checkpoint_dir "$CHECKPOINT_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --device "$DEVICE" \
            2>&1 | tee "$LOG_DIR/${source_city}_2_${target_city}.log"

        if [ $? -eq 0 ]; then
            echo "✓ $source_city → $target_city done"
        else
            echo "✗ $source_city → $target_city failed — check $LOG_DIR/${source_city}_2_${target_city}.log"
        fi
    done
done

echo "All predictions done."