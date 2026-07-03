import argparse
import psycopg2
import subprocess
import sys


DB_CONFIG = dict(
    host     = "localhost",
    port     = 5432,
    dbname   = "gis",
    user     = "gis",
    password = "gis",
)

def main(args):
    city = args.city
    
    sql = '''
        SELECT   name, 1 - (embedding <=> (SELECT embedding FROM cities WHERE name = :city_name)) AS similarity
        FROM     cities
        WHERE    name != :city_name
        ORDER BY embedding <=> (SELECT embedding FROM cities WHERE name = :city_name)
        LIMIT    1;
    '''
    
    conn = psycopg2.connect(**DB_CONFIG)
    
    with conn.cursor() as cur:
        cur.execute("""
            SELECT name
            FROM   cities
            WHERE  name != %s
            ORDER BY embedding <=> (SELECT embedding FROM cities WHERE name = %s)
            LIMIT 1
        """, (city, city))
        
        row = cur.fetchone()
        similar_city = row[0] if row else None
    
    print('Similar City is ', similar_city)
    sys.exit(0)
    
    result = subprocess.run(['python', '3_predict_on_graphs.py', '--source_city', similar_city, '--target_city', city], check=True)
    if result.returncode != 0:
        print("Output:", result.stdout.strip())
        sys.exit(result.returncode)
    
    print(f'Done Infering data using {similar_city} model on {city}')
        
    result = subprocess.run(['python', '4_insert_to_DB.py', '--source_city', similar_city, '--target_city', city], check=True)
    if result.returncode != 0:
        print("Output:", result.stdout.strip())
        sys.exit(result.returncode)
    
    print(f'Done Inserting to the DB the new data of {city}')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="graph2vec pipeline → pgvector")
    parser.add_argument("--city", required=True, help="City name")
    args = parser.parse_args()
    main(args)