from dataclasses import dataclass


@dataclass
class DataPipelineConfig:
    """Configuration for the Home Credit Data Pipeline."""

    data_dir: str
    sample_ratio: float
    cache_dir: str = ".cache"
    is_inference: bool = False
    artifact_dir: str = "models/artifacts"
    chunk_size: int = 200_000
    chunked_export: bool = True
