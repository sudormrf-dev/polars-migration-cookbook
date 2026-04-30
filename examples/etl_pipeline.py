"""Full ETL pipeline demo: synthetic sales data → lazy pipeline → aggregated report.

Demonstrates a production-style Polars lazy pipeline using the ``patterns``
package helpers.  All data is generated in-memory with Python stdlib — no
external dependencies required.

Pipeline stages
---------------
1. **Ingest** — simulate a scan of a Parquet file (lazy scan pattern)
2. **Validate** — filter out invalid/incomplete rows
3. **Enrich** — derive computed columns (revenue, discount_flag)
4. **Aggregate** — group_by region × category with multiple aggs
5. **Join** — attach region metadata from a lookup table
6. **Rank** — window function: rank products by revenue within each region
7. **Collect** — materialise results (streaming mode path shown)
8. **Report** — print a formatted summary table

Run::

    python examples/etl_pipeline.py
"""

from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from patterns.expressions import (
    AggFunction,
    WindowFunction,
    WindowSpec,
    build_aggregation,
    build_filter,
    build_window_expr,
)
from patterns.lazy_evaluation import (
    LazyPipelineConfig,
    PipelineOptimizer,
    QueryPlan,
    ScanType,
    build_lazy_pipeline,
)
from patterns.performance import (
    ParallelConfig,
    Timer,
    profile_memory,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SEED = 2024
_N_ORDERS = 5_000
_CATEGORIES = ["electronics", "clothing", "food", "books", "sports", "home"]
_REGIONS = ["north", "south", "east", "west"]
_REGION_METADATA = {
    "north": {"country": "CA", "timezone": "EST", "sales_tax": 0.13},
    "south": {"country": "US", "timezone": "CST", "sales_tax": 0.08},
    "east": {"country": "US", "timezone": "EST", "sales_tax": 0.07},
    "west": {"country": "US", "timezone": "PST", "sales_tax": 0.095},
}


# ---------------------------------------------------------------------------
# Stage 0: Data generation (simulates Parquet source)
# ---------------------------------------------------------------------------


@dataclass
class SalesRow:
    """Single sales order record."""

    order_id: int
    customer_id: int
    product: str
    category: str
    region: str
    unit_price: float
    quantity: int
    discount_pct: float  # 0.0 to 0.5
    year: int
    month: int
    is_returned: bool


def generate_sales_data(n: int = _N_ORDERS) -> list[SalesRow]:
    """Generate *n* synthetic sales records deterministically.

    Args:
        n: Number of rows to generate.

    Returns:
        List of :class:`SalesRow` instances.
    """
    rng = random.Random(_SEED)
    rows: list[SalesRow] = []
    for i in range(1, n + 1):
        rows.append(
            SalesRow(
                order_id=i,
                customer_id=rng.randint(1, 500),
                product=f"prod_{rng.randint(1, 50):03d}",
                category=rng.choice(_CATEGORIES),
                region=rng.choice(_REGIONS),
                unit_price=round(rng.uniform(2.0, 800.0), 2),
                quantity=rng.randint(1, 20),
                discount_pct=round(rng.choice([0.0, 0.0, 0.05, 0.1, 0.15, 0.2, 0.5]), 2),
                year=rng.choice([2022, 2023, 2024]),
                month=rng.randint(1, 12),
                is_returned=rng.random() < 0.04,  # ~4% return rate
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Stage 1: Lazy scan (simulated)
# ---------------------------------------------------------------------------


def stage_scan_and_plan() -> tuple[QueryPlan, LazyPipelineConfig]:
    """Build the lazy query plan (simulates pl.scan_parquet).

    Returns:
        Tuple of (QueryPlan, LazyPipelineConfig) ready to be rendered.
    """
    plan = QueryPlan(source="data/sales_raw.parquet", scan_type=ScanType.PARQUET)

    # Stage 2: Validate — remove returns and zero-quantity rows
    plan.add_filter("is_returned", "==", False)
    plan.add_filter("quantity", ">", 0)
    plan.add_filter("unit_price", ">", 0.0)

    # Stage 3: Enrich — select needed columns only (projection pushdown)
    plan.add_select(
        [
            "order_id",
            "customer_id",
            "product",
            "category",
            "region",
            "unit_price",
            "quantity",
            "discount_pct",
            "year",
            "month",
        ]
    )

    # Stage 4: Aggregate — region × category totals
    plan.add_group_by(
        ["region", "category"],
        (
            'pl.col("unit_price").mean().alias("avg_price"), '
            'pl.col("quantity").sum().alias("total_qty"), '
            '(pl.col("unit_price") * pl.col("quantity") * '
            '(1 - pl.col("discount_pct"))).sum().alias("net_revenue"), '
            'pl.col("order_id").n_unique().alias("order_count")'
        ),
    )

    # Stage 5: Sort by net_revenue descending for ranking
    plan.add_sort("net_revenue", descending=True)

    config = LazyPipelineConfig(
        predicate_pushdown=True,
        projection_pushdown=True,
        simplify_expression=True,
        slice_pushdown=True,
        streaming=False,
    )

    return plan, config


# ---------------------------------------------------------------------------
# Stage 2: Execute equivalent logic in Python stdlib
# ---------------------------------------------------------------------------


@dataclass
class AggRow:
    """Aggregated row for one region × category combination."""

    region: str
    category: str
    avg_price: float
    total_qty: int
    net_revenue: float
    order_count: int
    revenue_rank: int = 0  # filled in stage 5


def stage_filter_validate(rows: list[SalesRow]) -> list[SalesRow]:
    """Remove returned, zero-quantity, and zero-price rows.

    Args:
        rows: Raw sales rows.

    Returns:
        Validated subset.
    """
    return [r for r in rows if not r.is_returned and r.quantity > 0 and r.unit_price > 0.0]


def stage_enrich(rows: list[SalesRow]) -> list[dict[str, Any]]:
    """Add derived columns: revenue, discount_flag.

    Args:
        rows: Validated rows.

    Returns:
        List of enriched dicts.
    """
    enriched: list[dict[str, Any]] = []
    for r in rows:
        gross = r.unit_price * r.quantity
        revenue = round(gross * (1.0 - r.discount_pct), 2)
        enriched.append(
            {
                "order_id": r.order_id,
                "customer_id": r.customer_id,
                "product": r.product,
                "category": r.category,
                "region": r.region,
                "unit_price": r.unit_price,
                "quantity": r.quantity,
                "discount_pct": r.discount_pct,
                "year": r.year,
                "month": r.month,
                "gross_revenue": round(gross, 2),
                "net_revenue": revenue,
                "discount_flag": r.discount_pct > 0.0,
            }
        )
    return enriched


def stage_aggregate(enriched: list[dict[str, Any]]) -> list[AggRow]:
    """Group by region × category and compute aggregates.

    Args:
        enriched: Enriched order rows.

    Returns:
        One :class:`AggRow` per (region, category) group.
    """
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in enriched:
        groups[(row["region"], row["category"])].append(row)

    agg_rows: list[AggRow] = []
    for (region, category), group in groups.items():
        prices = [r["unit_price"] for r in group]
        agg_rows.append(
            AggRow(
                region=region,
                category=category,
                avg_price=round(statistics.mean(prices), 2),
                total_qty=sum(r["quantity"] for r in group),
                net_revenue=round(sum(r["net_revenue"] for r in group), 2),
                order_count=len({r["order_id"] for r in group}),
            )
        )
    return sorted(agg_rows, key=lambda r: r.net_revenue, reverse=True)


def stage_join_metadata(agg_rows: list[AggRow]) -> list[dict[str, Any]]:
    """Left-join aggregated rows with region metadata.

    Args:
        agg_rows: Aggregated rows.

    Returns:
        Enriched dicts with country, timezone, sales_tax appended.
    """
    joined: list[dict[str, Any]] = []
    for row in agg_rows:
        meta = _REGION_METADATA.get(row.region, {})
        joined.append(
            {
                "region": row.region,
                "category": row.category,
                "avg_price": row.avg_price,
                "total_qty": row.total_qty,
                "net_revenue": row.net_revenue,
                "order_count": row.order_count,
                "country": meta.get("country", "??"),
                "timezone": meta.get("timezone", "??"),
                "sales_tax": meta.get("sales_tax", 0.0),
                "revenue_after_tax": round(row.net_revenue * (1 - meta.get("sales_tax", 0.0)), 2),
                "revenue_rank": row.revenue_rank,
            }
        )
    return joined


def stage_rank_within_region(joined: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Assign revenue rank within each region (window function equivalent).

    Args:
        joined: Joined rows.

    Returns:
        Rows with ``revenue_rank`` populated.
    """
    # Group by region, sort by net_revenue desc, assign rank
    region_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in joined:
        region_groups[row["region"]].append(row)

    ranked: list[dict[str, Any]] = []
    for rows_in_region in region_groups.values():
        sorted_region = sorted(rows_in_region, key=lambda r: r["net_revenue"], reverse=True)
        for rank, row in enumerate(sorted_region, start=1):
            ranked.append({**row, "revenue_rank": rank})

    return sorted(ranked, key=lambda r: r["net_revenue"], reverse=True)


# ---------------------------------------------------------------------------
# Stage 6: Streaming simulation
# ---------------------------------------------------------------------------


def simulate_streaming_pipeline(raw_rows: list[SalesRow], chunk_size: int = 500) -> dict[str, Any]:
    """Simulate Polars streaming mode by processing data in chunks.

    In real Polars, ``collect(streaming=True)`` processes data in batches
    without loading the full dataset into RAM.  This function mimics the
    semantics for demonstration purposes.

    Args:
        raw_rows: All raw rows (the "file").
        chunk_size: Rows processed per streaming chunk.

    Returns:
        Dict with aggregated totals and chunk statistics.
    """
    n_chunks = math.ceil(len(raw_rows) / chunk_size)
    chunk_totals: list[float] = []
    chunk_counts: list[int] = []

    for i in range(n_chunks):
        chunk = raw_rows[i * chunk_size : (i + 1) * chunk_size]
        valid = [r for r in chunk if not r.is_returned and r.quantity > 0]
        revenue = sum(r.unit_price * r.quantity * (1 - r.discount_pct) for r in valid)
        chunk_totals.append(revenue)
        chunk_counts.append(len(valid))

    return {
        "n_chunks": n_chunks,
        "chunk_size": chunk_size,
        "total_revenue": round(sum(chunk_totals), 2),
        "total_valid_rows": sum(chunk_counts),
        "avg_chunk_revenue": round(statistics.mean(chunk_totals), 2),
    }


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------


def print_report(final_rows: list[dict[str, Any]], top_n: int = 15) -> None:
    """Print a formatted summary table of the ETL results.

    Args:
        final_rows: Ranked, joined aggregation rows.
        top_n: Number of rows to display.
    """
    print("\n" + "=" * 85)
    print("  ETL PIPELINE RESULTS — Top Region × Category by Net Revenue")
    print("=" * 85)
    header = f"{'Rank':>4}  {'Region':<8} {'Category':<12} {'Avg Price':>9} "
    header += f"{'Total Qty':>9} {'Net Revenue':>12} {'Orders':>7} {'Country':>7}"
    print(header)
    print("-" * 85)
    for row in final_rows[:top_n]:
        line = (
            f"{row['revenue_rank']:>4}  "
            f"{row['region']:<8} "
            f"{row['category']:<12} "
            f"${row['avg_price']:>8.2f} "
            f"{row['total_qty']:>9,} "
            f"${row['net_revenue']:>11,.2f} "
            f"{row['order_count']:>7,} "
            f"{row['country']:>7}"
        )
        print(line)
    print("=" * 85)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the full ETL pipeline demonstration."""
    print("\nPolars Migration Cookbook — ETL Pipeline Demo")
    print("=" * 50)

    # ---- Stage 0: Generate data ----
    print(f"\n[Stage 0] Generating {_N_ORDERS:,} synthetic orders...")
    with Timer("generate", rows=_N_ORDERS, operation="data_generation") as t_gen:
        raw_rows = generate_sales_data(_N_ORDERS)
    gen_result = t_gen.result
    assert gen_result is not None
    print(f"  Generated {len(raw_rows):,} rows in {gen_result.ms:.2f} ms")

    # ---- Stage 1: Show lazy pipeline plan ----
    print("\n[Stage 1] Building lazy query plan...")
    plan, config = stage_scan_and_plan()
    pipeline_expr = build_lazy_pipeline(plan, config)
    optimizer = PipelineOptimizer(plan, config)
    stats = optimizer.gather_stats()
    hints = optimizer.suggestions()
    print(
        f"  Steps: {stats.total_steps} "
        f"(filters={stats.filter_count}, projections={stats.projection_count}, "
        f"group_bys={stats.group_by_count}, sorts={stats.sort_count})"
    )
    print(f"  Active optimizations: {[f.value for f in config.active_optimizations()]}")
    if hints:
        print(f"  Optimizer hints: {hints}")
    print(f"  Pipeline expression (first 120 chars): {pipeline_expr[:120]}...")

    # ---- Stage 2: Filter & validate ----
    print("\n[Stage 2] Validating rows (remove returns, zero-qty)...")
    with Timer("validate", rows=len(raw_rows), operation="filter") as t_val:
        valid_rows = stage_filter_validate(raw_rows)
    val_result = t_val.result
    assert val_result is not None
    removed = len(raw_rows) - len(valid_rows)
    print(f"  {len(valid_rows):,} valid rows ({removed} removed) in {val_result.ms:.2f} ms")

    # Show filter expression for the validate step
    filter_expr = build_filter(
        [("is_returned", "==", False), ("quantity", ">", 0), ("unit_price", ">", 0.0)],
        combine="and",
    )
    print(f"  Polars filter expression: df.filter({filter_expr})")

    # ---- Stage 3: Enrich ----
    print("\n[Stage 3] Enriching rows (revenue, discount_flag)...")
    with Timer("enrich", rows=len(valid_rows), operation="with_columns") as t_en:
        enriched = stage_enrich(valid_rows)
    en_result = t_en.result
    assert en_result is not None
    print(f"  Enriched {len(enriched):,} rows in {en_result.ms:.2f} ms")

    # ---- Stage 4: Aggregate ----
    print("\n[Stage 4] Aggregating by region × category...")
    agg_expr = build_aggregation(
        group_by=["region", "category"],
        aggs={"net_revenue": AggFunction.SUM, "order_id": AggFunction.COUNT},
        aliases={"net_revenue": "total_revenue", "order_id": "order_count"},
    )
    with Timer("aggregate", rows=len(enriched), operation="group_by") as t_agg:
        agg_rows = stage_aggregate(enriched)
    agg_result = t_agg.result
    assert agg_result is not None
    print(f"  Aggregated into {len(agg_rows)} groups in {agg_result.ms:.2f} ms")
    print(f"  Polars aggregation: {agg_expr}")

    # ---- Stage 5: Join metadata ----
    print("\n[Stage 5] Joining region metadata...")
    joined = stage_join_metadata(agg_rows)
    print(f"  Joined {len(joined)} rows with region lookup")
    print("  Polars: df.join(region_meta_lf, on='region', how='left')")

    # ---- Stage 6: Window function — rank within region ----
    print("\n[Stage 6] Applying window function: rank by revenue within region...")
    spec = WindowSpec(partition_by=["region"], order_by="net_revenue", descending=True)
    window_expr = build_window_expr("net_revenue", WindowFunction.RANK, spec)
    with Timer("window_rank", rows=len(joined), operation="window") as t_win:
        ranked = stage_rank_within_region(joined)
    win_result = t_win.result
    assert win_result is not None
    print(f"  Ranked {len(ranked)} rows in {win_result.ms:.2f} ms")
    print(f"  Polars window expression: df.with_columns({window_expr})")

    # ---- Stage 7: Streaming simulation ----
    print("\n[Stage 7] Streaming simulation (chunk_size=500)...")
    with Timer("streaming", rows=len(raw_rows), operation="streaming") as t_stream:
        stream_stats = simulate_streaming_pipeline(raw_rows, chunk_size=500)
    stream_result = t_stream.result
    assert stream_result is not None
    print(f"  Processed {stream_stats['n_chunks']} chunks × {stream_stats['chunk_size']} rows")
    print(f"  Total valid rows: {stream_stats['total_valid_rows']:,}")
    print(f"  Total revenue (streaming): ${stream_stats['total_revenue']:,.2f}")
    print(f"  Streaming time: {stream_result.ms:.2f} ms")
    print("  Polars equivalent: lf.collect(streaming=True) or lf.sink_parquet('out.parquet')")

    # ---- Stage 8: Memory profiling ----
    print("\n[Stage 8] Memory profiling...")
    schema = {
        "order_id": "Int64",
        "customer_id": "Int32",
        "product": "Utf8",
        "category": "Categorical",
        "region": "Categorical",
        "unit_price": "Float64",
        "quantity": "Int32",
        "discount_pct": "Float32",
        "year": "Int16",
        "month": "Int8",
        "net_revenue": "Float64",
    }
    mem = profile_memory(schema, n_rows=_N_ORDERS)
    par_config = ParallelConfig(n_threads=8, use_streaming=False, chunk_size=500)
    print(f"  Estimated {_N_ORDERS:,}-row DataFrame: {mem.total_mb:.2f} MB")
    print(f"  Needs streaming (4 GB limit): {par_config.is_streaming_recommended(mem)}")
    print(f"  Largest columns: {mem.largest_columns(3)}")
    print(f"  ParallelConfig env vars: {par_config.env_vars()}")

    # ---- Final report ----
    print_report(ranked, top_n=12)

    # ---- Pipeline summary ----
    total_ms = sum(
        [
            gen_result.ms,
            val_result.ms,
            en_result.ms,
            agg_result.ms,
            win_result.ms,
            stream_result.ms,
        ]
    )
    print(f"\nPipeline total wall-time (Python stdlib): {total_ms:.2f} ms")
    print("(Real Polars lazy pipeline would be 10-100x faster via multi-threading + SIMD)")
    print("\nETL pipeline demo completed successfully.\n")


if __name__ == "__main__":
    main()
