import argparse
import sys
import os
sys.path.append(os.path.expanduser("~/websites/mapedia"))
from modules import DBHandler, DBUpdater
import pandas as pd
import numpy as np
from modules.db_handler.DBConfig import INTER_CITY_LEARNING_SOURCE, INTRA_CITY_LEARNING_SOURCE, EMPTY_SOURCE
PRESET_CONFIDENCE = 0.9
PRESET_CONFIDENCE = 0.1
import psycopg2
from tqdm import tqdm
import io

def connect_to_db_psycopg2():
    """Initializes a pure psycopg2 connection if it doesn't exist."""
    conn = psycopg2.connect(
        "postgresql://gis:gis@cs-u-spatial-406.cs.umn.edu:5432/gis"
    )
    return conn

def convert_to_speed_matrix(speed_metadata_df):
    ''' takes as input the datafrmae that contains the columns of the proposed dynamic vector values and output a numpy array of the vector'''
    
    # Period label → (start_hour, end_hour)
    PERIOD_HOURS = {
        "00-04": range(0, 4),
        "04-08": range(4, 8),
        "08-12": range(8, 12),
        "12-16": range(12, 16),
        "16-20": range(16, 20),
        "20-24": range(20, 24),
    }

    # weekday=0 means Monday in pandas; 0-4 = weekday, 5-6 = weekend
    WEEKDAY_DAYS = list(range(5))   # 0-4
    WEEKEND_DAYS = list(range(5, 7)) # 5-6

    # 1. One period-block: 6 periods × their hour counts = 24 values
    #    Each period value repeated for its hours
    period_cols_weekday = [f"pred_avg_speed_weekday_{p}" for p in PERIOD_HOURS]
    period_cols_weekend = [f"pred_avg_speed_weekend_{p}" for p in PERIOD_HOURS]
    period_lengths      = [len(h) for h in PERIOD_HOURS.values()]  # [4,4,4,4,4,4]

    def build_day_vector(row, cols):
        """24 values: each period value repeated for its hour count"""
        return np.repeat([row[c] for c in cols], period_lengths)  # (24,)

    def build_week_vector(row):
        """168 values: 5× weekday-day + 2× weekend-day"""
        day = build_day_vector(row, period_cols_weekday)  # (24,)
        end = build_day_vector(row, period_cols_weekend)  # (24,)
        return np.concatenate([np.tile(day, 5), np.tile(end, 2)])  # (168,)

    def build_speed_array(row):
        """672 values: week vector repeated 4× for seasons"""
        return np.tile(build_week_vector(row), 4)  # (672,)

    # Apply once per road
    speed_matrix = np.stack(speed_metadata_df.apply(build_speed_array, axis=1))  # (N, 672)
    
    return speed_matrix

def update_old_vals(old_val, old_source, old_conf, new_val, new_source, new_conf):
    print('3. Compute the boolean mask across the entire 2D grid instantly')
    empty_old_val = np.isnan(old_val)
    better_source = new_source < old_source
    same_source_better_conf = (new_source == old_source) & (new_conf >= old_conf)

    # Combine conditions into a single master mask
    mask = empty_old_val | better_source | same_source_better_conf

    print('4. Apply changes to the arrays in-place where the mask is True')
    old_val[mask] = new_val[mask]
    old_source[mask] = new_source[mask]
    old_conf[mask] = new_conf[mask]
    
    return old_val, old_source, old_conf

def upload_to_database(conn, ordered_ids, speed_matrix, speed_matrix_source, speed_matrix_conf):

    # 2. Set Up Chunking
    print('Set Up Chunking')
    # Over 1 billion array elements will cause RAM bloat if written to a single stream.
    chunk_size = 50000
    total_rows = len(ordered_ids)

    # Open a single cursozr context manager
    with conn.cursor() as cur:

        # 3. Create High-Performance UNLOGGED Staging Table
        print('Create High-Performance UNLOGGED Staging Table')
        # ON COMMIT DROP is replaced with manual dropping since pure psycopg2 transactions vary
        cur.execute("""
            DROP TABLE IF EXISTS tmp_road_update;
            CREATE UNLOGGED TABLE tmp_road_update (
                id bigint,
                avg_speed numeric[],
                avg_speed_source integer[],
                avg_speed_conf double precision[]
            );
        """)

        # 4. Stream Data in Chunks
        print('Stream Data in Chunks')
        for i in range(0, total_rows, chunk_size):
            end_idx = min(i + chunk_size, total_rows)

            buffer = io.StringIO()

            # Fast string formatting loop for the active chunk
            for j in tqdm(range(i, end_idx), total=(end_idx-i), desc=f'Processing Chunk{i}'):
                # Format into native PostgreSQL text array strings: {val1,val2,val3}
                # print('pgspeed conversion to string')
                pg_speed = (
                    str(speed_matrix[j].tolist())
                    .replace("[", "{")
                    .replace("]", "}")
                )
                # print('pgconf to string')
                pg_conf = (
                    str(speed_matrix_conf[j].tolist())
                    .replace("[", "{")
                    .replace("]", "}")
                )

                # For the integer array, we safely cast numbers to int if they aren't NaN.
                # Keeping them as "nan" maps safely to database NULLs later.
                # print('pgsource to string')
                pg_source = (
                    "{"
                    + ",".join(
                        [
                            str(int(x)) if not np.isnan(x) else "NULL"
                            for x in speed_matrix_source[j]
                        ]
                    )
                    + "}"
                )

                # Write tab-separated row straight to memory buffer
                # print('writing to buffer')
                buffer.write(
                    f"{int(ordered_ids[j])}\t{pg_speed}\t{pg_source}\t{pg_conf}\n"
                )

            buffer.seek(0)

            # Stream data chunk straight into the staging table bypassing SQL engine parsing
            print('Copying from the buffer to the table directly')
            cur.copy_from(
                buffer,
                "tmp_road_update",
                columns=("id", "avg_speed", "avg_speed_source", "avg_speed_conf"),
                null="NULL",
            )
            buffer.close()

        # 5. Index the Staging Table
        # Speeds up the final JOIN execution on 500,000 rows drastically.
        print('Creating Index on the temp_road_update')
        cur.execute("CREATE INDEX idx_tmp_road_id ON tmp_road_update(id);")

        # 6. Execute Single Native Bulk Update Join
        print('Execute Update Join')
        cur.execute("""
            UPDATE road_attributes r
            SET 
                avg_speed = t.avg_speed,
                avg_speed_source = t.avg_speed_source,
                avg_speed_conf = t.avg_speed_conf
            FROM tmp_road_update t
            WHERE r.id = t.id
        """)

        # 7. Cleanup Staging Table
        # print('Droping the newly created table')
        # cur.execute("DROP TABLE tmp_road_update;")

        # Commit everything to the database at once
        print('Commiting')
        conn.commit()

def insert(metadata, db_handler, source):
    print('Taking the static metadata')
    pred_metadata = metadata[['mapd_id', 'osm_id',
        'pred_road_type', 'pred_nlanes_cls',
       'pred_oneway', 'pred_width', 'pred_max_speed', 'pred_min_speed']].copy()
    pred_metadata.rename(columns={
        'pred_road_type': 'road_type', 
        'pred_nlanes_cls': 'nlanes',
        'pred_oneway': 'oneway', 
        'pred_width': 'width', 
        'pred_max_speed': 'max_speed', 
        'pred_min_speed': 'min_speed'
    }, inplace=True)
    print('setting their confidence and source')
    for c in ['road_type', 'nlanes', 'oneway', 'width', 'max_speed', 'min_speed']:
        pred_metadata[f'{c}_source'] = source
        pred_metadata[f'{c}_conf'] = PRESET_CONFIDENCE
    
    
    # Generating the Speed Matrix, Source and Confidence matricies as numpy arrays
    print('Taking the Speed dynamic metadata')
    speed_metadata_df = metadata[['mapd_id', 'osm_id',
       'pred_avg_speed_weekday_00-04',
       'pred_avg_speed_weekday_04-08',
       'pred_avg_speed_weekday_08-12',
       'pred_avg_speed_weekday_12-16',
       'pred_avg_speed_weekday_16-20',
       'pred_avg_speed_weekday_20-24',
       'pred_avg_speed_weekend_00-04',
       'pred_avg_speed_weekend_04-08',
       'pred_avg_speed_weekend_08-12',
       'pred_avg_speed_weekend_12-16',
       'pred_avg_speed_weekend_16-20',
       'pred_avg_speed_weekend_20-24']].copy()
    print('Converting them into matricies')
    speed_matrix  = convert_to_speed_matrix(speed_metadata_df)
    speed_matrix_source = np.full(speed_matrix.shape, source, dtype=np.float32)
    speed_matrix_source[np.isnan(speed_matrix)] = np.nan
    speed_matrix_conf   = np.full(speed_matrix.shape, PRESET_CONFIDENCE, dtype=np.float32)
    speed_matrix_conf[np.isnan(speed_matrix)] = np.nan
    new_val = speed_matrix
    new_source = speed_matrix_source
    new_conf = speed_matrix_conf
    
    # Generating the 
    print('Getting old values from the database')
    ordered_ids = pred_metadata.mapd_id.to_numpy()
    road_attributes = db_handler.roads_from_ids(ordered_ids.tolist())
    
    print('Converting them into matricies')
    road_attr = road_attributes.loc[ordered_ids]
    old_val = np.array(road_attr["avg_speed"].tolist(), dtype=np.float32)
    old_source = np.array(road_attr["avg_speed_source"].tolist(), dtype=np.float32)
    old_conf = np.array(road_attr["avg_speed_conf"].tolist(), dtype=np.float32)

    print('Masking the new values to the old ones')
    val, source, conf = update_old_vals(old_val, old_source, old_conf, new_val, new_source, new_conf)

    print('connecting to the database')
    conn = connect_to_db_psycopg2()
    
    print('inserting to database the dynamic part')
    upload_to_database(conn, ordered_ids, val, source, conf)
    
    print('inserting to database the static part')
    db_updater = DBUpdater(db_handler)
    static_metadata = pred_metadata.set_index("mapd_id").rename_axis(None)
    db_updater.update_database_new(static_attr=static_metadata)
    
    print('done insertion')
    
    

def parse_args():
    p = argparse.ArgumentParser(description="Insert metadata into the database")
    p.add_argument("--source_city",            default="jakarta")
    p.add_argument("--target_city",            default="jakarta")
    p.add_argument("--data_dir",        default="./data/imputed_data")
    # p.add_argument("--file", default="jakarta_imputedBy_jakarta.parquet")
    # p.add_argument("--checkpoint_dir",  default="./checkpoints")
    # p.add_argument("--input",          default='./data/imputed_data/jakarta.parquet',
                #    help="Output parquet path. Defaults to <data_dir>/<city>_imputed.parquet")
    # p.add_argument("--device",          default="cuda",
    #                help="'auto', 'cpu', 'cuda', 'cuda:0', …")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    filename = f"{args.target_city}_imputedBy_{args.source_city}.parquet"
    metadata_path  = os.path.join(args.data_dir, filename)
    metadata = pd.read_parquet(metadata_path)
    
    source = INTRA_CITY_LEARNING_SOURCE if args.target_city == args.source_city else INTER_CITY_LEARNING_SOURCE
    db_handler = DBHandler()
    db_handler.connect_to_db()
    
    insert(metadata, db_handler, source)