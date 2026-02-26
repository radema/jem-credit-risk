import polars as pl
import os
from pathlib import Path

cache_dir = "/home/rauldemaio/projects_local/jem-credit-risk/.cache"
if not os.path.exists(cache_dir):
    print(f"{cache_dir} does not exist.")
    exit()

query_col = "numberofoverdueinstlmaxdat_641D"
files = list(Path(cache_dir).glob("test_*.parquet"))
for f in sorted(files):
    try:
        # scan a single row to get schema
        lf = pl.scan_parquet(str(f))
        schema = lf.collect_schema()
        if query_col in schema:
            print(f"FILE: {f.name} | COLUMN: {query_col} | DTYPE: {schema[query_col]}")
            # check the actual data if it's Null
            # df = lf.select(query_col).collect()
            # print(f"Sample: {df[query_col].head(5).to_list()}")
    except Exception as e:
        pass
