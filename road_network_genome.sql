CREATE TYPE region_level AS ENUM (
    'country',
    'state',
    'county',
    'tract',
    'block_group',
    'block',
    'custom_cell'
);


CREATE TABLE regions (
    region_id          BIGSERIAL PRIMARY KEY,
    external_id        TEXT NOT NULL,
    external_source    TEXT NOT NULL DEFAULT 'US_CENSUS',
    region_name        TEXT NOT NULL,
    region_type        region_level NOT NULL,

    parent_region_id   BIGINT REFERENCES regions(region_id),
    hierarchy_level    SMALLINT NOT NULL,

    area_size_km2      DOUBLE PRECISION,
    geom               GEOMETRY(MULTIPOLYGON, 4326),

    UNIQUE (external_source, region_type, external_id)
);


CREATE TABLE genomes (
    region_id          BIGINT NOT NULL REFERENCES regions(region_id),
    time_bin           TEXT NOT NULL,
    genome_uri         TEXT,
    static_embedding   VECTOR(256),
    full_embedding     VECTOR(256),
    genome_status      TEXT NOT NULL DEFAULT 'pending',

    PRIMARY KEY (region_id, time_bin)
);