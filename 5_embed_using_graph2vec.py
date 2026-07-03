#!/usr/bin/env python3
"""
5_embed_using_graph2vec.py

Runs the full graph2vec pipeline for a set of cities and stores
the resulting embeddings into a PostgreSQL pgvector table.

Usage:
    python run_graph2vec_pipeline.py --cities cairo london tokyo --dimensions 128
    python run_graph2vec_pipeline.py \
    --cities cairo london tokyo singapore \
    --dimensions 128 \
    --epochs 50 \
    --workers 4
    
    python 5_embed_using_graph2vec.py \
    --cities jakarta singapore chicago NewYorkCity sanFrancisco washingtonDC 
"""

import argparse
import subprocess
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR        = Path("./data/raw_data")
CITIES_JSON     = Path("./cities.json")
INPUT_DIR       = Path("./embedding_models/data/graph2vec/input")
OUTPUT_DIR      = Path("./embedding_models/data/graph2vec/output")
EMBEDDINGS_CSV  = OUTPUT_DIR / "embeddings_named.csv"
PLOT_DIR        = OUTPUT_DIR #Path("./graph_embeddings")

DB_CONFIG = dict(
    host     = "localhost",
    port     = 5432,
    dbname   = "gis",
    user     = "gis",
    password = "gis",
)

TABLE_NAME = "city_trained_models"

TABLE_NAME = "cities"

# ── Helpers ───────────────────────────────────────────────────────────────────

def run(cmd: list[str], step: str) -> None:
    print(f"\n{'─'*60}")
    print(f"[{step}] {' '.join(cmd)}")
    print('─'*60)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"[ERROR] Step '{step}' failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def filter_cities_json(cities: list[str], src: Path, dst: Path) -> None:
    """Write a filtered cities.json containing only the requested cities."""
    with open(src) as f:
        all_cities = json.load(f)

    # Support both list-of-dicts (with "name" key) and dict-of-dicts
    if isinstance(all_cities, list):
        filtered = [c for c in all_cities if c["name"] in cities]
    else:
        filtered = {k: v for k, v in all_cities.items() if k in cities}

    if not filtered:
        print(f"[ERROR] None of {cities} found in {src}")
        sys.exit(1)

    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w") as f:
        json.dump(filtered, f, indent=2)
    print(f"[INFO] Filtered cities.json → {dst}  ({len(filtered)} cities)")

def upsert_embeddings(conn, df: pd.DataFrame) -> None: 
    dim_cols = [c for c in df.columns if c not in ("city", "graph_id")]

    rows = []
    for _, row in df.iterrows():
        city_name = row["city"]
        vec = row[dim_cols].to_numpy(dtype=np.float32).tolist()
        rows.append((city_name, vec))

    with conn.cursor() as cur:
        execute_values(cur, f"""
            UPDATE public.{TABLE_NAME}
            SET    embedding = data.embedding::vector
            FROM (VALUES %s) AS data(name, embedding)
            WHERE  {TABLE_NAME}.name = data.name;
        """,
        rows,  # list of (name, embedding) tuples
        template="(%s, %s::vector)"
        )
    conn.commit()
    print(f"[DB] Updated embeddings for {len(rows)} cities in '{TABLE_NAME}'")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="graph2vec pipeline → pgvector")
    parser.add_argument("--cities",      nargs="+", required=True,
                        help="City names to process (must match cities.json keys)")
    parser.add_argument("--dimensions",  type=int,  default=128)
    parser.add_argument("--wl-iterations", type=int, default=3)
    parser.add_argument("--epochs",      type=int,  default=50)
    parser.add_argument("--workers",     type=int,  default=4)
    parser.add_argument("--feature-mode", default="attributed",
                        choices=["attributed", "degree"])
    parser.add_argument("--skip-embed",  action="store_true",
                        help="Skip steps 1-2, only re-upload existing embeddings CSV")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # Filter cities.json to requested subset
    filtered_json = OUTPUT_DIR / "cities_filtered.json"
    filter_cities_json(args.cities, CITIES_JSON, filtered_json)

    if not args.skip_embed:
        # ── Step 1: Prepare ───────────────────────────────────────────────────
        run([
            sys.executable, "embedding_models/graph2vec_support.py",
            "--mode",         "prepare",
            "--feature_mode", args.feature_mode,
            "--data_dir",     str(DATA_DIR),
            "--cities_json",  str(filtered_json),
            "--input_dir",    str(INPUT_DIR),
        ], step="1/3 prepare")

        # ── Step 2: Embed ─────────────────────────────────────────────────────
        run([
            sys.executable, "embedding_models/graph2vec/src/graph2vec.py",
            "--input-path",   str(INPUT_DIR),
            "--output-path",  str(EMBEDDINGS_CSV),
            "--dimensions",   str(args.dimensions),
            "--wl-iterations",str(args.wl_iterations),
            "--epochs",       str(args.epochs),
            "--workers",      str(args.workers),
        ], step="2/3 graph2vec")

        # ── Step 3: Postprocess ───────────────────────────────────────────────
        run([
            sys.executable, "embedding_models/graph2vec_support.py",
            "--mode",        "postprocess",
            "--cities_json", str(filtered_json),
            "--input_dir",   str(INPUT_DIR),
            "--output_csv",  str(EMBEDDINGS_CSV),
            "--plot_dir",    str(PLOT_DIR),
        ], step="3/3 postprocess")

    # ── Step 4: Load into pgvector ────────────────────────────────────────────
    EMBEDDINGS_CSV_NAMED = EMBEDDINGS_CSV.with_stem(EMBEDDINGS_CSV.stem + "_named")

    if not EMBEDDINGS_CSV_NAMED.exists():
        print(f"[ERROR] Embeddings CSV not found: {EMBEDDINGS_CSV_NAMED}")
        sys.exit(1)

    df = pd.read_csv(EMBEDDINGS_CSV_NAMED)
    print(f"[INFO] Loaded embeddings: {df.shape}  columns: {list(df.columns[:5])} ...")

    if "city" not in df.columns:
        print("[ERROR] Postprocessed CSV must have a 'city' column")
        sys.exit(1)

    # Filter to only requested cities in case CSV has more
    df = df[df["city"].isin(args.cities)].reset_index(drop=True)
    print(f"[INFO] Filtered to {len(df)} requested cities")

    conn = psycopg2.connect(**DB_CONFIG)
    try:
        upsert_embeddings(conn, df)
    finally:
        conn.close()

    print("\n✓ Pipeline complete.")


if __name__ == "__main__":
    main()