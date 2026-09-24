#!/bin/bash
# Same source/target cross-prediction sweep as cross_predict_all_cities.sh,
# but parallelized across GPUs (NUM_GPUS x PROCS_PER_GPU concurrent jobs,
# round-robin over "cuda:$gpu_idx"), following train_all_cities.sh's job-pool
# pattern. city_grids.json has ~326 tile-level regions, so the full source x
# target sweep is ~326*326 pairs — this is what makes parallelizing worth it.

CITIES_JSON="./city_grids.json"
DATA_DIR="./data/raw_data"
CHECKPOINT_DIR="./checkpoints"
OUTPUT_DIR="./data/imputed_data"
RESULTS_DIR="./results"
LOG_DIR="./logs/infer"
EPOCHS=3000

NUM_GPUS=8
PROCS_PER_GPU=2
NUM_SLOTS=$(( NUM_GPUS * PROCS_PER_GPU ))

mkdir -p "$OUTPUT_DIR" "$RESULTS_DIR" "$LOG_DIR"

cities=$(python3 -c "import json; print('\n'.join(json.load(open('$CITIES_JSON')).keys()))")
cities_array=($cities)
total=${#cities_array[@]}
total_pairs=$(( total * total ))

echo "Found $total regions in $CITIES_JSON -> $total_pairs source/target pairs"
echo "Running with $NUM_GPUS GPUs x $PROCS_PER_GPU proc/GPU = $NUM_SLOTS parallel jobs"

declare -a pids
declare -a pid_labels

slot=0
launched=0

for source_city in "${cities_array[@]}"; do
    for target_city in "${cities_array[@]}"; do
        gpu_idx=$(( slot % NUM_GPUS ))
        slot=$(( slot + 1 ))
        launched=$(( launched + 1 ))
        label="${source_city}_2_${target_city}"

        python 3_predict_on_graphs.py \
            --source_city "$source_city" \
            --target_city "$target_city" \
            --data_dir "$DATA_DIR" \
            --checkpoint_dir "$CHECKPOINT_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --device "cuda:$gpu_idx" \
            --epochs $EPOCHS \
            > "$LOG_DIR/${label}.log" 2>&1 &

        pids+=($!)
        pid_labels+=("$label")

        # Only wait for a slot once the pool is full
        if [ ${#pids[@]} -ge $NUM_SLOTS ]; then
            wait "${pids[0]}"
            status=$?
            done_label="${pid_labels[0]}"

            if [ $status -eq 0 ]; then
                echo "✓ [$launched/$total_pairs] $done_label done"
            else
                echo "✗ [$launched/$total_pairs] $done_label failed — check $LOG_DIR/${done_label}.log"
            fi

            pids=("${pids[@]:1}")
            pid_labels=("${pid_labels[@]:1}")
        fi
    done
done

# Wait for remaining jobs (handles the tail end of the queue)
echo "Waiting for remaining jobs to finish..."
for i in "${!pids[@]}"; do
    wait "${pids[$i]}"
    status=$?
    done_label="${pid_labels[$i]}"

    if [ $status -eq 0 ]; then
        echo "✓ $done_label done"
    else
        echo "✗ $done_label failed — check $LOG_DIR/${done_label}.log"
    fi
done

echo "=========================================="
echo "All predictions done."
