import polars as pl
import logging

logger = logging.getLogger(__name__)


def handle_missing_and_categoricals(df: pl.DataFrame) -> pl.DataFrame:
    """
    Handles missing values and categorical features according to the following logic:
    - Categorical variables: Replace each with integer frequency (count frequency encoding).
    - Missing Categoricals: Fill NaN with "MISSING" string literal. Treat as its own category.
    - Missing Continuous: Median Imputation + Missing Indicator (binary column).
    """
    logger.info("Handling missing values and categorical features...")

    exclude_cols = ["case_id", "WEEK_NUM", "target", "num_group1", "num_group2"]

    numeric_cols = []
    cat_cols = []

    schema = df.schema
    for col, dtype in schema.items():
        if col in exclude_cols:
            continue

        if dtype in [
            pl.Int8,
            pl.Int16,
            pl.Int32,
            pl.Int64,
            pl.UInt8,
            pl.UInt16,
            pl.UInt32,
            pl.UInt64,
            pl.Float32,
            pl.Float64,
            pl.Decimal,
        ]:
            numeric_cols.append(col)
        elif dtype in [pl.String, pl.Categorical, pl.Boolean]:
            cat_cols.append(col)
        else:
            logger.warning(f"Unrecognized dtype {dtype} for column {col}. Skipping.")

    logger.info(
        f"Processing {len(numeric_cols)} numeric columns and {len(cat_cols)} categorical columns."
    )

    # 1. Numeric: create missing indicator exprs
    null_exprs = []
    fill_num_exprs = []

    for col in numeric_cols:
        col_dtype = schema[col]

        # indicator expr
        if col_dtype in [pl.Float32, pl.Float64]:
            is_miss_expr = (
                (pl.col(col).is_null() | pl.col(col).is_nan())
                .cast(pl.Int8)
                .alias(f"{col}_is_null")
            )
        else:
            is_miss_expr = pl.col(col).is_null().cast(pl.Int8).alias(f"{col}_is_null")
        null_exprs.append(is_miss_expr)

        # fill expr
        fill_expr = pl.col(col).fill_null(pl.col(col).median())
        if col_dtype in [pl.Float32, pl.Float64]:
            fill_expr = fill_expr.fill_nan(pl.col(col).median())

        fill_num_exprs.append(fill_expr.alias(col))

    # 2. Categorical / Boolean: fill with "MISSING"
    fill_cat_exprs = [
        pl.col(col).cast(pl.String).fill_null("MISSING").alias(col) for col in cat_cols
    ]

    # Combine step A: indicators and filled features
    df = df.with_columns(null_exprs + fill_num_exprs + fill_cat_exprs)

    # 3. Categorical: map to frequencies
    freq_exprs = [
        pl.col(col).count().over(col).cast(pl.Int32).alias(col) for col in cat_cols
    ]

    # Combine step B: rewrite cat columns to counts
    if cat_cols:
        df = df.with_columns(freq_exprs)

    logger.info("Completed handling of missing values and categoricals.")
    return df
