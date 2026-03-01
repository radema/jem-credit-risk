import os

from src.data.config import DataPipelineConfig
from src.data.pipeline import run_pipeline
from src.model.jem.data_utils import get_dataloaders, split_data_chronologically

# Change working directory so relative paths work from project root
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def main():
    print("Generating feature_cols.json...")
    cfg = DataPipelineConfig(
        data_dir="data/raw/home-credit-credit-risk-model-stability.zip",
        sample_ratio=0.1,
        cache_dir=".cache",
    )
    df_full = run_pipeline(
        config=cfg,
        depth_0_tables=["train_static_0", "train_static_cb_0"],
        depth_1_tables=[
            "train_person_1",
            "train_deposit",
            "train_debitcard",
            "train_other",
        ],
        depth_2_tables=[],
    )
    df_train, df_val = split_data_chronologically(df_full, val_weeks=12)

    # This call runs the low-variance filter deterministically and drops columns.
    # It will save feature_cols.json to models/artifacts/
    train_loader, val_loader_for_eval, scaler, feature_cols = get_dataloaders(
        df_train, df_val, batch_size=256, artifact_dir="models/artifacts"
    )
    print("feature_cols.json successfully generated in models/artifacts/")


if __name__ == "__main__":
    main()
