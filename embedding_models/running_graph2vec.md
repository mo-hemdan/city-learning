# Running Graph2Vec on our input & Ouput

### Step 1: Prepare Input JSONs:

```bash
python graph2vec_support.py \
    --mode prepare \
    --data_dir ../data/raw_data \
    --cities_json ../cities.json \
    --feature_mode attributed
    --input_dir ./data/graph2vec/input
```

### Step 2 — Clone and run graph2vec:

```bash
python graph2vec/src/graph2vec.py \
    --input-path  ./data/graph2vec/input \
    --output-path ./data/graph2vec/output/embeddings.csv \
    --dimensions 128 \
    --wl-iterations 3 \
    --epochs 50 \
    --workers 4
```

### Step 3 — Postprocess: attach city names + plot:

```bash
python graph2vec_support.py \
    --mode postprocess \
    --input_dir   ./data/graph2vec/input \
    --output_csv  ./data/graph2vec/output/embeddings.csv \
    --plot_dir    ./data/graph2vec/output
```
