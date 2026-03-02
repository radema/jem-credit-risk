import argparse
import logging

import numpy as np
import polars as pl
import torch

from src.data.config import DataPipelineConfig
from src.data.pipeline import run_pipeline
from src.model.jem.config import JEMConfig
from src.model.jem.data_utils import get_inference_dataloader, load_feature_cols
from src.model.jem.infer import load_inference_pipeline, perform_inference

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("InferencePipeline")


def main():
    parser = argparse.ArgumentParser(description="Generate Home Credit Submission")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw/home-credit-credit-risk-model-stability.zip",
        help="Path to the competition data (zip or directory).",
    )
    parser.add_argument(
        "--artifact-dir",
        type=str,
        default="models/artifacts",
        help="Directory containing trained models and imputer state.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="submission.csv",
        help="Where to save the final submission CSV.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4096, help="Batch size for inference."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run inference on (cpu, cuda, mps).",
    )

    args = parser.parse_args()
    device = torch.device(args.device)

    # 1. Pipeline Execution (Phase 2)
    # We use the Archetype tables as defined in the pipeline orchestration
    depth_0_tables = ["train_static_0", "train_static_cb_0"]
    depth_1_tables = ["train_credit_bureau_a_1", "train_person_1"]
    depth_2_tables = ["train_person_2"]

    logger.info("--- Phase 1: Running Data Pipeline in INFERENCE mode ---")
    cfg = DataPipelineConfig(
        data_dir=args.data_dir,
        sample_ratio=1.0,  # Process all test samples
        is_inference=True,
        artifact_dir=args.artifact_dir,
    )

    # run_pipeline returns the collected Polars DataFrame
    test_df = run_pipeline(
        config=cfg,
        depth_0_tables=depth_0_tables,
        depth_1_tables=depth_1_tables,
        depth_2_tables=depth_2_tables,
    )

    feature_cols = load_feature_cols(args.artifact_dir)
    logger.info(f"Identified {len(feature_cols)} feature columns for inference.")

    # 2. Loading Models (Phase 1)
    # We assume standard latent_dim=64 and num_classes=2 for JEM
    logger.info("--- Phase 2: Loading Pre-trained Models ---")
    jem_config = JEMConfig(artifact_dir=args.artifact_dir)

    scaler, ae, jem = load_inference_pipeline(
        artifact_dir=args.artifact_dir,
        input_dim=len(feature_cols),
        latent_dim=64,
        num_classes=2,
        jem_config=jem_config,
        device=device,
    )

    # 3. Batch Inference (Phase 3)
    logger.info("--- Phase 3: Executing Batched Inference ---")
    loader = get_inference_dataloader(test_df, feature_cols, batch_size=args.batch_size)

    case_ids, probs, energies = perform_inference(
        scaler=scaler,
        autoencoder=ae,
        jem=jem,
        loader=loader,
        device=device,
        return_energies=True,
    )

    # 4. Submission Generation (Phase 4)
    logger.info("--- Phase 4: Finalizing Submission ---")
    submission_df = pl.DataFrame({"case_id": case_ids, "score": probs})

    # Kaggle requires specifically 'case_id' and 'score' columns
    submission_df.write_csv(args.output_path)
    logger.info(
        f"Submission saved to {args.output_path} (Shape: {submission_df.shape})"
    )

    # 5. Diagnostic Reporting (Task 2)
    if energies is not None:
        mean_e = np.mean(energies)
        std_e = np.std(energies)
        logger.info(f"Diagnostic: Test Energy Mean = {mean_e:.4f}, Std = {std_e:.4f}")

        # Simple heuristic: if energy is extremely high, the samples might be OOD
        # In a real scenario, we would compare this against training energy stats stored in artifacts.
        logger.info("Diagnostic complete. Pipeline finished successfully.")


if __name__ == "__main__":
    main()
