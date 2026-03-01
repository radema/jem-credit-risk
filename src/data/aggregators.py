import logging

import polars as pl

logger = logging.getLogger(__name__)


def _get_agg_expressions(schema, depth: str):
    """
    Returns a list of Polars expressions for aggregation depending on column types.
    """
    aggs = []

    # Columns to omit from general aggregations
    keys = ["case_id", "num_group1", "num_group2"]

    for col_name, dtype in schema.items():
        if col_name in keys:
            continue

        # Handle join flags specifically to avoid renaming/multiple aggs
        if col_name.endswith("_joined_flag"):
            # For join flags, just take the max (which is 1) and keep the name
            aggs.append(pl.col(col_name).max().alias(col_name))
            continue

        is_numeric = dtype in [
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
        ]

        if depth == "depth_2":
            # For Depth 2, keep it simple to avoid feature name explosion
            # (e.g. col_max_max) when aggregated again in Depth 1.
            if is_numeric:
                aggs.append(pl.col(col_name).max().alias(col_name))
            else:
                aggs.append(pl.col(col_name).last().alias(col_name))
        else:
            # Depth 1: full advanced aggregations
            if is_numeric:
                aggs.append(pl.col(col_name).max().alias(f"{col_name}_max"))
                aggs.append(pl.col(col_name).min().alias(f"{col_name}_min"))
                aggs.append(pl.col(col_name).mean().alias(f"{col_name}_mean"))
                aggs.append(pl.col(col_name).std().alias(f"{col_name}_std"))
                iqr = pl.col(col_name).quantile(0.75) - pl.col(col_name).quantile(0.25)
                aggs.append(iqr.alias(f"{col_name}_iqr"))
            else:
                # Categorical
                aggs.append(pl.col(col_name).last().alias(f"{col_name}_last"))
                aggs.append(pl.col(col_name).first().alias(f"{col_name}_first"))

                mode_expr = pl.col(col_name).mode().last()
                aggs.append(mode_expr.alias(f"{col_name}_mode"))

                # Counts
                aggs.append(
                    pl.col(col_name).filter(pl.col(col_name) == mode_expr).count().alias(f"{col_name}_mode_count")
                )
                aggs.append(
                    pl.col(col_name).filter(pl.col(col_name) == pl.col(col_name).last()).count().alias(f"{col_name}_last_count")
                )
                aggs.append(
                    pl.col(col_name).filter(pl.col(col_name) == pl.col(col_name).first()).count().alias(f"{col_name}_first_count")
                )
                aggs.append(pl.col(col_name).mode().count().alias(f"{col_name}_n_modes"))

    return aggs


def aggregate_depth_2(lazy_df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Aggregates a depth-2 table (which has multiple num_group2 per num_group1)
    down to depth-1 (one row per case_id and num_group1).
    """
    # Force schema resolution for dynamic evaluation
    schema = lazy_df.collect_schema()

    aggs = _get_agg_expressions(schema, depth="depth_2")

    if not aggs:
        logger.info("No columns to aggregate in depth-2 table.")
        return lazy_df.select(["case_id", "num_group1"]).unique()

    return lazy_df.group_by(["case_id", "num_group1"]).agg(aggs)


def aggregate_depth_1(lazy_df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Aggregates a depth-1 table (or a previously aggregated depth-2 table)
    down to depth-0 (one row per case_id).
    """
    schema = lazy_df.collect_schema()

    aggs = _get_agg_expressions(schema, depth="depth_1")

    if not aggs:
        logger.info("No columns to aggregate in depth-1 table.")
        return lazy_df.select(["case_id"]).unique()

    return lazy_df.group_by("case_id").agg(aggs)


def join_to_base(base_lazy: pl.LazyFrame, dict_of_lazy_dfs: dict) -> pl.LazyFrame:
    """
    Left-joins all flattened LazyFrames onto the base LazyFrame.
    """
    final_df = base_lazy
    for table_name, table_lazy in dict_of_lazy_dfs.items():
        logger.info(f"Joining {table_name} onto base...")
        final_df = final_df.join(table_lazy, on="case_id", how="left")

    return final_df
