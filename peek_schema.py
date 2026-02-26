import polars as pl
import os
from pathlib import Path

cache_dir = ".cache"
if not os.path.exists(cache_dir):
    print(f"{cache_dir} does not exist.")
    exit()

files = list(Path(cache_dir).glob("test_*.parquet"))
for f in sorted(files):
    try:
        schema = pl.scan_parquet(str(f)).collect_schema()
        if "numberofoverdueinstlmaxdat_641D" in schema:
            print(f"{f.name}: {schema['numberofoverdueinstlmaxdat_641D']}")
    except Exception as e:
        print(f"Error scanning {f}: {e}")
