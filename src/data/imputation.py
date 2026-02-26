import polars as pl
import logging

logger = logging.getLogger(__name__)


def handle_missing_and_categoricals(
    df: pl.DataFrame, state: dict | None = None
) -> tuple[pl.DataFrame, dict]:
    """
    Handles missing values and categorical features according to the following logic:
    - Categorical variables: Replace each with integer frequency (count frequency encoding).
    - Missing Categoricals: Fill NaN with "MISSING" string literal. Treat as its own category.
    - Missing Continuous: Median Imputation + Missing Indicator (binary column).
    """
    logger.info("Handling missing values and categorical features...")

    is_fitting = state is None
    if is_fitting:
        state = {"medians": {}, "freqs": {}}

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

    if is_fitting and numeric_cols:
        medians_dict = df.select(
            [pl.col(c).median().alias(c) for c in numeric_cols]
        ).to_dicts()[0]
        # Handle cases where median might be None (all nulls)
        state["medians"] = {
            c: (m if m is not None else 0.0) for c, m in medians_dict.items()
        }

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
        median_val = state["medians"][col]
        fill_expr = pl.col(col).fill_null(median_val)
        if col_dtype in [pl.Float32, pl.Float64]:
            fill_expr = fill_expr.fill_nan(median_val)

        fill_num_exprs.append(fill_expr.alias(col))

    # 2. Categorical / Boolean: fill with "MISSING"
    fill_cat_exprs = [
        pl.col(col).cast(pl.String).fill_null("MISSING").alias(col) for col in cat_cols
    ]

    # Combine step A: indicators and filled features
    df = df.with_columns(null_exprs + fill_num_exprs + fill_cat_exprs)

    # 3. Categorical: map to frequencies
    if is_fitting:
        for col in cat_cols:
            vc = df[col].value_counts()
            c1, c2 = vc.columns[0], vc.columns[1]
            # Convert to dictionary (mapping string to count)
            freq_map = dict(zip(vc[c1].to_list(), vc[c2].to_list()))
            state["freqs"][col] = freq_map

    freq_exprs = []
    for col in cat_cols:
        freq_map = state["freqs"][col]
        col_expr = pl.col(col)
        # Using strict typing or casting based on polars version
        if hasattr(pl.Expr, "replace"):
            expr = col_expr.replace(freq_map, default=1).cast(pl.Int32).alias(col)
        else:
            expr = col_expr.map_dict(freq_map, default=1).cast(pl.Int32).alias(col)
        freq_exprs.append(expr)

    # Combine step B: rewrite cat columns to counts
    if cat_cols:
        df = df.with_columns(freq_exprs)

    if is_fitting:
        state["final_columns"] = df.columns
    else:
        # Schema Alignment: Ensure all columns from training are present
        train_cols = state.get("final_columns", [])
        if train_cols:
            missing_cols = [c for c in train_cols if c not in df.columns]
            if missing_cols:
                logger.info(
                    f"Adding {len(missing_cols)} missing columns to the test set."
                )
                # We'll fill with 0 or MISSING based on the col name/state
                fill_values = {}
                for c in missing_cols:
                    if c in state["medians"]:
                        fill_values[c] = state["medians"][c]
                    else:
                        fill_values[c] = 0  # Default fallback

                df = df.with_columns(
                    [pl.lit(v).alias(c) for c, v in fill_values.items()]
                )

            # Reorder columns to match training and drop any extra columns in test
            df = df.select(train_cols)

    logger.info("Completed handling of missing values and categoricals.")
    return df, state
