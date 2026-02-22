import polars as pl
import numpy as np

df = pl.DataFrame(
    {
        "case_id": [1, 2, 3, 4, 5],
        "target": [0, 1, 0, 1, 0],
        "num_feature": [1.0, np.nan, 3.0, None, 5.0],
        "cat_feature": ["A", "A", None, "B", "B"],
        "bool_feature": [True, False, None, True, False],
    }
)

numeric_cols = ["num_feature"]
cat_cols = ["cat_feature", "bool_feature"]


# 1. create is_null features (handling both NaN and Null)
# To handle both, we should create the indicator first
# Polars has `.is_null()` and `.is_nan()`. is_nan is only valid for float columns.
# We can use .is_null() | .is_nan() for Float columns, but for Int columns is_nan is not available.
def get_is_missing(col):
    if df[col].dtype in [pl.Float32, pl.Float64]:
        return (
            (pl.col(col).is_null() | pl.col(col).is_nan())
            .cast(pl.Int8)
            .alias(f"{col}_is_null")
        )
    return pl.col(col).is_null().cast(pl.Int8).alias(f"{col}_is_null")


def fill_missing_num(col):
    expr = pl.col(col).fill_null(pl.col(col).median())
    if df[col].dtype in [pl.Float32, pl.Float64]:
        expr = expr.fill_nan(pl.col(col).median())
    return expr


null_exprs = [get_is_missing(col) for col in numeric_cols]

fill_num_exprs = [fill_missing_num(col) for col in numeric_cols]

fill_cat_exprs = [
    pl.col(col).cast(pl.String).fill_null("MISSING").alias(col) for col in cat_cols
]

df_filled = df.with_columns(null_exprs + fill_num_exprs + fill_cat_exprs)

freq_exprs = [
    pl.col(col).count().over(col).cast(pl.Int32).alias(col) for col in cat_cols
]

df_encoded = df_filled.with_columns(freq_exprs)
print(df_encoded)
