import os
import zipfile
import logging

logger = logging.getLogger(__name__)


def extract_relevant_parquets(zip_path: str, cache_dir: str = ".cache") -> str:
    """
    Extracts strictly the 'train_' and 'test_' parquet files from the Kaggle dataset zip.
    If the environment is Kaggle (e.g. /kaggle/input is detected) or the path is a directory,
    extraction is skipped.

    Returns the path to the directory containing the parquet files to scan.
    """
    if os.path.isdir(zip_path) or "kaggle" in zip_path.lower():
        logger.info(
            f"Skipping extraction. Assuming files are natively accessible at {zip_path}"
        )
        return zip_path

    if not zip_path.endswith(".zip"):
        logger.warning(f"Expected a .zip file, got {zip_path}. Using as directory.")
        return zip_path

    os.makedirs(cache_dir, exist_ok=True)

    logger.info(
        f"Extracting matching 'train_'/'test_' parquets from {zip_path} to {cache_dir}"
    )
    extracted_count = 0
    with zipfile.ZipFile(zip_path, "r") as archive:
        all_files = archive.namelist()
        # Filter for train/test parquet files
        relevant_parquets = [
            f
            for f in all_files
            if f.endswith(".parquet")
            and ("train_" in os.path.basename(f) or "test_" in os.path.basename(f))
        ]

        for file in relevant_parquets:
            target_path = os.path.join(cache_dir, os.path.basename(file))
            if not os.path.exists(target_path):
                # We need to extract the specific file and write it flat to the cache_dir
                logger.debug(f"Extracting {file} to {target_path}")
                with archive.open(file) as source, open(target_path, "wb") as target:
                    target.write(source.read())
                extracted_count += 1

    logger.info(f"Successfully extracted {extracted_count} matching files.")
    return cache_dir
