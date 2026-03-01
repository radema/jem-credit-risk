import polars as pl

df = pl.DataFrame(
    {
        "a": [1, 2, None],
        "b": [1.1, None, 3.3],
        "c": ["str1", "str2", "str3"],
        "target": [0, 1, 0],
        "case_id": [1, 2, 3],
        "MONTH": [1, 1, 1],
        "WEEK_NUM": [1, 2, 3],
    }
)

exclude_cols = ["case_id", "MONTH", "WEEK_NUM", "target"]
feature_cols = [
    col for col in df.columns if col not in exclude_cols and df[col].dtype.is_numeric()
]

df_train = df.with_columns(
    pl.col(feature_cols).cast(pl.Float32).fill_null(0.0).fill_nan(0.0)
)
print(
    "Numeric subset extracted correctly. Shape:",
    df_train.select(feature_cols).to_numpy().shape,
)
