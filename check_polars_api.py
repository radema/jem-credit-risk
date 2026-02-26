import polars as pl
import inspect

try:
    sig = inspect.signature(pl.scan_parquet)
    print(f"Polars Version: {pl.__version__}")
    print("Arguments for pl.scan_parquet:")
    for param in sig.parameters:
        print(f" - {param}")
except Exception as e:
    print(f"Error: {e}")
