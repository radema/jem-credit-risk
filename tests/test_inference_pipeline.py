import pytest
import polars as pl
import pickle
from pathlib import Path
from src.data.pipeline import run_pipeline
from src.data.config import DataPipelineConfig


@pytest.fixture
def mock_data_dir(tmp_path):
    data_dir = tmp_path / "raw_data"
    data_dir.mkdir()
    return str(data_dir)


@pytest.fixture
def mock_artifact_dir(tmp_path):
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    return str(artifact_dir)


def test_inference_pipeline_end_to_end(mock_data_dir, mock_artifact_dir, monkeypatch):
    # 1. Setup Mock Tables
    # Training Base: case_id, target, feature1
    train_base = pl.DataFrame(
        {
            "case_id": [1, 2, 3],
            "target": [0, 1, 0],
            "feature1": [10.0, 20.0, None],  # One null for imputation
        }
    )

    # Test Base: case_id, feature1 (NO target)
    test_base = pl.DataFrame({"case_id": [4, 5], "feature1": [30.0, None]})

    # Mock scan_table to return these
    def mock_scan(table_name, cache_dir):
        if "train_base" in table_name:
            return train_base.lazy()
        if "test_base" in table_name:
            return test_base.lazy()
        return pl.DataFrame({"case_id": []}).lazy()

    monkeypatch.setattr("src.data.pipeline.scan_table", mock_scan)
    monkeypatch.setattr("src.data.pipeline.extract_relevant_parquets", lambda x, y: x)
    # Mock aggregators and export since we focus on pipeline flow and imputation
    monkeypatch.setattr("src.data.pipeline.aggregate_depth_1", lambda x: x)
    monkeypatch.setattr("src.data.pipeline.aggregate_depth_2", lambda x: x)
    monkeypatch.setattr("src.data.pipeline.evaluate_eda_stats", lambda x, y: None)
    monkeypatch.setattr("src.data.pipeline.export_to_parquet", lambda x, y: None)

    # --- PHASE 1: TRAINING ---
    train_cfg = DataPipelineConfig(
        data_dir=mock_data_dir,
        sample_ratio=1.0,
        artifact_dir=mock_artifact_dir,
        is_inference=False,
    )

    _ = run_pipeline(train_cfg, [], [], [])

    # Verify artifact created
    state_path = Path(mock_artifact_dir) / "imputer_state.pkl"
    assert state_path.exists()

    with open(state_path, "rb") as f:
        state = pickle.load(f)

    assert "medians" in state
    assert state["medians"]["feature1"] == 15.0  # median of 10 and 20
    assert "final_columns" in state
    assert "feature1" in state["final_columns"]

    # --- PHASE 2: INFERENCE ---
    test_cfg = DataPipelineConfig(
        data_dir=mock_data_dir,
        sample_ratio=1.0,
        artifact_dir=mock_artifact_dir,
        is_inference=True,
    )

    # We'll also test schema alignment by creating a test set missing 'feature1'
    # but having an extra unexpected 'feature2'
    test_base_missing = pl.DataFrame(
        {
            "case_id": [6],
            "feature2": [999.0],  # Extra column, should be dropped
        }
    )

    def mock_scan_inf(table_name, cache_dir):
        if "test_base" in table_name:
            return test_base_missing.lazy()
        return pl.DataFrame({"case_id": []}).lazy()

    monkeypatch.setattr("src.data.pipeline.scan_table", mock_scan_inf)

    test_out = run_pipeline(test_cfg, [], [], [])

    # Verify Output
    assert "target" not in test_out.columns
    assert "feature1" in test_out.columns  # Added via schema alignment
    assert "feature2" not in test_out.columns  # Dropped via schema alignment
    assert test_out["feature1"][0] == 15.0  # Imputed with training median
    assert test_out["case_id"][0] == 6

    print("Inference Pipeline Test Passed!")
