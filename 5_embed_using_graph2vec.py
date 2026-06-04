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
EMBEDDINGS_CSV  = OUTPUT_DIR / "embeddings.csv"
PLOT_DIR        = OUTPUT_DIR #Path("./graph_embeddings")

DB_CONFIG = dict(
    host     = "localhost",
    port     = 5432,
    dbname   = "gis",
    user     = "gis",
    password = "gis",
)

TABLE_NAME = "city_trained_models"

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


def ensure_table(conn, dimensions: int) -> None:
    """Add embedding column to city_trained_models if not present."""
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(f"""
            ALTER TABLE public.city_trained_models
                ADD COLUMN IF NOT EXISTS embedding     vector({dimensions}),
                ADD COLUMN IF NOT EXISTS embedding_dim integer;
                
        """) # 
    conn.commit()
    print(f"[DB] Embedding columns ready on '{TABLE_NAME}' (dim={dimensions})")

def get_city_ids(conn, city_names: list[str]) -> dict[str, int]:
    """Returns {city_name: city_id} from the cities table."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, name FROM public.cities WHERE name = ANY(%s)",
            (city_names,)
        )
        rows = cur.fetchall()

    found = {name: cid for cid, name in rows}
    missing = set(city_names) - set(found)
    if missing:
        print(f"[WARN] Cities not found in DB: {missing}")
    return found


def upsert_embeddings(conn, df: pd.DataFrame, city_ids: dict[str, int], dimensions: int) -> None:
    dim_cols = [c for c in df.columns if c not in ("city", "graph_id")]

    rows = []
    for _, row in df.iterrows():
        city_name = row["city"]
        city_id   = city_ids.get(city_name)
        if city_id is None:
            print(f"[SKIP] '{city_name}' not found in cities table")
            continue
        vec = row[dim_cols].to_numpy(dtype=np.float32).tolist()
        rows.append((vec, dimensions, city_id))

    with conn.cursor() as cur:
        execute_values(cur, f"""
            UPDATE public.{TABLE_NAME}
            SET    embedding     = data.embedding::vector,
                   embedding_dim = data.dim,
                   trained_at    = now()
            FROM (VALUES %s) AS data(embedding, dim, city_id)
            WHERE  {TABLE_NAME}.city_id = data.city_id;
        """,
        rows,
        template="(%s::vector, %s, %s)"
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
            sys.executable, "prepare_graph2vec_input.py",
            "--mode",         "prepare",
            "--feature_mode", args.feature_mode,
            "--data_dir",     str(DATA_DIR),
            "--cities_json",  str(filtered_json),
            "--input_dir",    str(INPUT_DIR),
        ], step="1/3 prepare")

        # ── Step 2: Embed ─────────────────────────────────────────────────────
        run([
            sys.executable, "graph2vec/src/graph2vec.py",
            "--input-path",   str(INPUT_DIR),
            "--output-path",  str(EMBEDDINGS_CSV),
            "--dimensions",   str(args.dimensions),
            "--wl-iterations",str(args.wl_iterations),
            "--epochs",       str(args.epochs),
            "--workers",      str(args.workers),
        ], step="2/3 graph2vec")

        # ── Step 3: Postprocess ───────────────────────────────────────────────
        run([
            sys.executable, "prepare_graph2vec_input.py",
            "--mode",        "postprocess",
            "--cities_json", str(filtered_json),
            "--input_dir",   str(INPUT_DIR),
            "--output_csv",  str(EMBEDDINGS_CSV),
            "--plot_dir",    str(PLOT_DIR),
        ], step="3/3 postprocess")

    # ── Step 4: Load into pgvector ────────────────────────────────────────────
    if not EMBEDDINGS_CSV.exists():
        print(f"[ERROR] Embeddings CSV not found: {EMBEDDINGS_CSV}")
        sys.exit(1)

    df = pd.read_csv(EMBEDDINGS_CSV)
    print(f"[INFO] Loaded embeddings: {df.shape}  columns: {list(df.columns[:5])} ...")

    if "city" not in df.columns:
        print("[ERROR] Postprocessed CSV must have a 'city' column")
        sys.exit(1)

    # Filter to only requested cities in case CSV has more
    df = df[df["city"].isin(args.cities)].reset_index(drop=True)
    print(f"[INFO] Filtered to {len(df)} requested cities")

    conn = psycopg2.connect(**DB_CONFIG)
    try:
        ensure_table(conn, dimensions=args.dimensions)
        city_ids = get_city_ids(conn, args.cities)
        upsert_embeddings(conn, df, city_ids, dimensions=args.dimensions)
    finally:
        conn.close()

    print("\n✓ Pipeline complete.")


if __name__ == "__main__":
    main()