#!/bin/bash

CITIES_JSON="./cities.json"
LOG_DIR="./logs"
NUM_GPUS=8

mkdir -p "$LOG_DIR"

cities=$(python -c "import json; print('\n'.join(json.load(open('$CITIES_JSON')).keys()))")
cities_array=($cities)
total=${#cities_array[@]}

echo "Found $total cities, distributing across $NUM_GPUS GPUs"

declare -a pids
declare -a pid_cities
declare -a pid_gpus

gpu_idx=0

for city in "${cities_array[@]}"; do
    echo "Launching: $city on GPU $gpu_idx"

    python 2_train_on_graphs.py --city "$city" --device "cuda:$gpu_idx" \
        > "$LOG_DIR/${city}.log" 2>&1 &

    pids+=($!)
    pid_cities+=("$city")
    pid_gpus+=("$gpu_idx")

    gpu_idx=$(( (gpu_idx + 1) % NUM_GPUS ))

    # Only wait for a slot if we've filled all GPUs
    if [ ${#pids[@]} -ge $NUM_GPUS ]; then
        wait "${pids[0]}"
        status=$?

        city_done="${pid_cities[0]}"
        gpu_done="${pid_gpus[0]}"

        if [ $status -eq 0 ]; then
            echo "✓ $city_done (GPU $gpu_done) completed successfully"
        else
            echo "✗ $city_done (GPU $gpu_done) failed — check $LOG_DIR/${city_done}.log"
        fi

        pids=("${pids[@]:1}")
        pid_cities=("${pid_cities[@]:1}")
        pid_gpus=("${pid_gpus[@]:1}")
    fi
done

# Wait for remaining jobs (this handles cases where total cities < NUM_GPUS)
echo "Waiting for remaining jobs to finish..."
for i in "${!pids[@]}"; do
    wait "${pids[$i]}"
    status=$?

    city_done="${pid_cities[$i]}"
    gpu_done="${pid_gpus[$i]}"

    if [ $status -eq 0 ]; then
        echo "✓ $city_done (GPU $gpu_done) completed successfully"
    else
        echo "✗ $city_done (GPU $gpu_done) failed — check $LOG_DIR/${city_done}.log"
    fi
done

echo "=========================================="
echo "All cities done."