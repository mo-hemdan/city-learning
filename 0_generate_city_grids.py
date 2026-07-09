import json
import math

CITIES_FILE = './cities.json'
OUTPUT_FILE = './city_grids.json'
CELL_SIZE_KM = 5
KM_PER_DEG_LAT = 111.32


def km_per_deg_lon(lat_deg):
    return KM_PER_DEG_LAT * math.cos(math.radians(lat_deg))


def make_grid(bounds, cell_size_km=CELL_SIZE_KM):
    min_lat, max_lat = bounds['min_lat'], bounds['max_lat']
    min_lon, max_lon = bounds['min_lon'], bounds['max_lon']

    mid_lat = (min_lat + max_lat) / 2
    lat_step = cell_size_km / KM_PER_DEG_LAT
    lon_step = cell_size_km / km_per_deg_lon(mid_lat)

    n_rows = max(1, math.ceil((max_lat - min_lat) / lat_step))
    n_cols = max(1, math.ceil((max_lon - min_lon) / lon_step))

    areas = {}
    for row in range(n_rows):
        area_min_lat = min_lat + row * lat_step
        area_max_lat = min(area_min_lat + lat_step, max_lat)
        for col in range(n_cols):
            area_min_lon = min_lon + col * lon_step
            area_max_lon = min(area_min_lon + lon_step, max_lon)
            areas[f'r{row}_c{col}'] = {
                'min_lat': area_min_lat,
                'max_lat': area_max_lat,
                'min_lon': area_min_lon,
                'max_lon': area_max_lon,
            }
    return areas


with open(CITIES_FILE, 'r') as f:
    city_bounds = json.load(f)

city_grids = {}
for city, bounds in city_bounds.items():
    areas = make_grid(bounds)
    for area_name, area_bounds in areas.items():
        city_grids[f'{city}_{area_name}'] = area_bounds
    print(f'{city}: {len(areas)} areas')

with open(OUTPUT_FILE, 'w') as f:
    json.dump(city_grids, f, indent=2)

print(f'Saved grids to {OUTPUT_FILE}')