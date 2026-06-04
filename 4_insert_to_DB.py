"""
insert.py  –  Upload Predicted values to the database

Usage
-----
    python insert.py \
        --city jakarta \
        --data_dir ./data/raw_data \
        --checkpoint_dir ./checkpoints \
        --output ./data/raw_data/jakarta_imputed.parquet
"""
import sys
import argparse
import os
sys.path.append(os.path.expanduser("~/websites/mapedia"))
from modules import DBHandler, DBUpdater
import pandas as pd


# db_updater.update_database(static_attr=result, static_cols=['nlanes', 'width', 'oneway', 'road_type'])


def insert(metadata, db_updater):
    print(metadata.head())
    print(metadata.columns)
    
    pass
    
    
    

def parse_args():
    p = argparse.ArgumentParser(description="Impute missing road attributes with trained MultiAttrGAT")
    p.add_argument("--source_city",            default="jakarta")
    p.add_argument("--target_city",            default="jakarta")
    p.add_argument("--data_dir",        default="./data/imputed_data")
    # p.add_argument("--checkpoint_dir",  default="./checkpoints")
    # p.add_argument("--input",          default='./data/imputed_data/jakarta.parquet',
                #    help="Output parquet path. Defaults to <data_dir>/<city>_imputed.parquet")
    # p.add_argument("--device",          default="cuda",
    #                help="'auto', 'cpu', 'cuda', 'cuda:0', …")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    metadata_path  = os.path.join(args.data_dir, f"{args.target_city}_imputedBy_{args.source_city}.parquet")
    # speed_path  = os.path.join(args.data_dir, f"{args.target_city}_speed_matrix.npy")
    # output_path = args.output or os.path.join(args.data_dir, f"{args.target_city}_imputedBy_{args.source_city}.parquet")
    
    metadata = pd.read_parquet(metadata_path)
    
    db_handler = DBHandler()
    db_handler.connect_to_db()

    db_updater = DBUpdater(db_handler)
    
    insert(
        metadata        = metadata,
        db_updater      = db_updater,
    )