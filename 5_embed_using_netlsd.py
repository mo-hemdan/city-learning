#!/usr/bin/env python3
"""
5_embed_using_netlsd.py

Runs the full NetLSD pipeline for a set of cities and stores the resulting
spectral graph descriptors into a PostgreSQL pgvector table.

Unlike 5_embed_using_graph2vec.py, NetLSD needs no training step: it computes
a fixed-length descriptor directly from each city's road-segment line-graph
Laplacian spectrum (heat or wave kernel trace). It is purely structural and
does not use road attributes (highway type, speed, ...) — only the segment
adjacency topology.

Usage:
    python 5_embed_using_netlsd.py --cities_json ./cities.json
    python 5_embed_using_netlsd.py \
        --cities_json ./cities.json \
        --kernel      heat \
        --dimensions  128

The cities to process are taken from the keys of --cities_json, e.g.:
    {
      "jakarta": {...},
      "singapore": {...}
    }
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

DATA_DIR       = Path("./data/raw_data")
OUTPUT_DIR     = Path("./embedding_models/data/netlsd/output")
EMBEDDINGS_CSV = OUTPUT_DIR / "embeddings_named.csv"
PLOT_DIR       = OUTPUT_DIR

DB_CONFIG = dict(
    host     = "localhost",
    port     = 5432,
    dbname   = "gis",
    user     = "gis",
    password = "gis",
)

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


def load_cities(cities_json: Path) -> list[str]:
    """Cities to process are the top-level keys of cities_json."""
    with open(cities_json) as f:
        cities = list(json.load(f).keys())

    if not cities:
        print(f"[ERROR] No cities found in {cities_json}")
        sys.exit(1)

    print(f"[INFO] Loaded {len(cities)} cities from {cities_json}: {cities}")
    return cities


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
    parser = argparse.ArgumentParser(description="NetLSD pipeline → pgvector")
    parser.add_argument("--cities_json", type=Path, required=True,
                        help="Path to a JSON file whose top-level keys are the city names to process")
    parser.add_argument("--kernel",      choices=["heat", "wave"], default="heat")
    parser.add_argument("--dimensions",  type=int, default=128,
                        help="Descriptor length (must match the pgvector column width, 128)")
    parser.add_argument("--normalization", default="empty", choices=["empty", "complete", "none"])
    parser.add_argument("--skip-embed",  action="store_true",
                        help="Skip steps 1-2, only re-upload existing embeddings CSV")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cities = load_cities(args.cities_json)

    if not args.skip_embed:
        # ── Step 1: Compute descriptors ─────────────────────────────────────────
        run([
            sys.executable, "embedding_models/netlsd_support.py",
            "--mode",          "compute",
            "--data_dir",      str(DATA_DIR),
            "--cities_json",   str(args.cities_json),
            "--output_csv",    str(EMBEDDINGS_CSV),
            "--kernel",        args.kernel,
            "--dimensions",    str(args.dimensions),
            "--normalization", args.normalization,
        ], step="1/2 compute")

        # ── Step 2: Postprocess ───────────────────────────────────────────────
        run([
            sys.executable, "embedding_models/netlsd_support.py",
            "--mode",       "postprocess",
            "--output_csv", str(EMBEDDINGS_CSV),
            "--plot_dir",   str(PLOT_DIR),
        ], step="2/2 postprocess")

    # ── Step 3: Load into pgvector ────────────────────────────────────────────
    if not EMBEDDINGS_CSV.exists():
        print(f"[ERROR] Embeddings CSV not found: {EMBEDDINGS_CSV}")
        sys.exit(1)

    df = pd.read_csv(EMBEDDINGS_CSV)
    print(f"[INFO] Loaded embeddings: {df.shape}  columns: {list(df.columns[:5])} ...")

    if "city" not in df.columns:
        print("[ERROR] Embeddings CSV must have a 'city' column")
        sys.exit(1)

    # Filter to only requested cities in case CSV has more
    df = df[df["city"].isin(cities)].reset_index(drop=True)
    print(f"[INFO] Filtered to {len(df)} requested cities")

    conn = psycopg2.connect(**DB_CONFIG)
    try:
        upsert_embeddings(conn, df)
    finally:
        conn.close()

    print("\n✓ Pipeline complete.")


if __name__ == "__main__":
    main()
