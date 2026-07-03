#!/bin/bash

CITIES_JSON="./cities.json"
DATA_DIR="./data/raw_data"
PYG_DATA_DIR="./data/pyg_data"
CHECKPOINT_DIR="./checkpoints"
OUTPUT_DIR="./data/imputed_data_finetuned"
RESULTS_DIR="./results/finetuned"
PLOTS_DIR="./plots/cross-city-finetuned/"
LOG_DIR="./logs/infer_finetuned"
DEVICE="cpu"

FT_EPOCHS=150
FT_LR=1e-4
FT_P_MASK=0.30
FT_PATIENCE=20

mkdir -p "$OUTPUT_DIR" "$RESULTS_DIR" "$PLOTS_DIR" "$LOG_DIR"

cities=$(python3 -c "import json; print('\n'.join(json.load(open('$CITIES_JSON')).keys()))")

for source_city in $cities; do
    for target_city in $cities; do
        echo "=========================================="
        echo "  source=$source_city  →  target=$target_city  (with fine-tuning)"
        echo "=========================================="

        python 3a_predict_on_graphs_withFinetuning.py \
            --source_city "$source_city" \
            --target_city "$target_city" \
            --data_dir "$DATA_DIR" \
            --pyg_data_dir "$PYG_DATA_DIR" \
            --checkpoint_dir "$CHECKPOINT_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --results_dir "$RESULTS_DIR" \
            --plots_dir "$PLOTS_DIR" \
            --ft_epochs "$FT_EPOCHS" \
            --ft_lr "$FT_LR" \
            --ft_p_mask "$FT_P_MASK" \
            --ft_patience "$FT_PATIENCE" \
            --device "$DEVICE" \
            2>&1 | tee "$LOG_DIR/${source_city}_2_${target_city}.log"

        if [ $? -eq 0 ]; then
            echo "✓ $source_city → $target_city (finetuned) done"
        else
            echo "✗ $source_city → $target_city (finetuned) failed — check $LOG_DIR/${source_city}_2_${target_city}.log"
        fi
    done
done

echo "All fine-tuned predictions done."
