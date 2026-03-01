import polars as pl
import pytest
from pathlib import Path
from src.data.export import export_to_chunked_parquet


def test_chunked_export_creates_correct_files(tmp_path):
    # Given a 500-row DataFrame and chunk_size=200
    df = pl.DataFrame(
        {
            "case_id": range(500),
            "a": range(500),
            "b": range(500),
            "month": [202401] * 500,
            "week": [1] * 500,
        }
    )

    chunk_dir = tmp_path / "chunks"
    chunk_size = 200
    prefix = "test_chunk"

    # When
    paths = export_to_chunked_parquet(
        df, str(chunk_dir), chunk_size=chunk_size, prefix=prefix
    )

    # Then
    assert len(paths) == 3
    assert (chunk_dir / f"{prefix}_001.parquet").exists()
    assert (chunk_dir / f"{prefix}_002.parquet").exists()
    assert (chunk_dir / f"{prefix}_003.parquet").exists()

    # Verify row counts
    assert pl.read_parquet(paths[0]).height == 200
    assert pl.read_parquet(paths[1]).height == 200
    assert pl.read_parquet(paths[2]).height == 100

    # Verify all case_ids are present
    combined_df = pl.concat([pl.read_parquet(p) for p in paths])
    assert combined_df.height == 500
    assert combined_df["case_id"].to_list() == list(range(500))
