"""Pandas → Polars migration demonstration: 8+ equivalent operations side-by-side.

This script walks through the most common pandas operations and shows the
equivalent Polars pattern using the expression/lazy/compat helpers from the
``patterns`` package.  All data is generated with Python stdlib — no pandas or
Polars install required.  The output shows the *expression strings* that would
be executed in a real Polars project, along with the result of running the
equivalent logic on pure-Python data structures.

Run::

    python examples/migration_demo.py
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

from patterns.expressions import (
    AggFunction,
    ExpressionBuilder,
    WindowFunction,
    WindowSpec,
    build_aggregation,
    build_filter,
    build_window_expr,
)
from patterns.lazy_evaluation import (
    LazyPipelineConfig,
    QueryPlan,
    ScanType,
    build_lazy_pipeline,
)
from patterns.pandas_compat import (
    ConversionConfig,
    DataFrameConverter,
    SchemaMapper,
)
from patterns.performance import (
    PerformanceAdvisor,
    Timer,
    profile_memory,
)

# ---------------------------------------------------------------------------
# Synthetic data generation (stdlib only)
# ---------------------------------------------------------------------------

_RANDOM_SEED = 42
_CATEGORIES = ["electronics", "clothing", "food", "books", "sports"]
_REGIONS = ["north", "south", "east", "west"]


def make_sales_rows(n: int = 200) -> list[dict[str, Any]]:
    """Generate *n* synthetic sales rows deterministically.

    Returns:
        List of dicts with keys: order_id, customer_id, product, category,
        region, amount, quantity, year, month.
    """
    rng = random.Random(_RANDOM_SEED)
    rows: list[dict[str, Any]] = []
    for i in range(1, n + 1):
        rows.append(
            {
                "order_id": i,
                "customer_id": rng.randint(1, 40),
                "product": f"prod_{rng.randint(1, 20):03d}",
                "category": rng.choice(_CATEGORIES),
                "region": rng.choice(_REGIONS),
                "amount": round(rng.uniform(5.0, 500.0), 2),
                "quantity": rng.randint(1, 10),
                "year": rng.choice([2022, 2023, 2024]),
                "month": rng.randint(1, 12),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Helper: pretty-print a section header
# ---------------------------------------------------------------------------


def _header(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def _show(label: str, pandas_code: str, polars_code: str, result: Any) -> None:
    """Print a side-by-side comparison card."""
    print(f"\n--- {label} ---")
    print(f"  [pandas]  {pandas_code}")
    print(f"  [polars]  {polars_code}")
    print(f"  result  → {result}")


# ---------------------------------------------------------------------------
# Operation 1: Simple column filter
# ---------------------------------------------------------------------------


def demo_filter(rows: list[dict[str, Any]]) -> None:
    """Show filter pattern: pandas boolean indexing → Polars filter expression."""
    _header("1. Filter rows (amount > 200 AND category == 'electronics')")

    # Equivalent pure-Python logic
    filtered = [r for r in rows if r["amount"] > 200.0 and r["category"] == "electronics"]

    # Build Polars expression string via patterns.expressions
    expr = build_filter(
        [("amount", ">", 200.0), ("category", "==", "electronics")],
        combine="and",
    )

    _show(
        "Filter",
        "df[(df['amount'] > 200) & (df['category'] == 'electronics')]",
        f"df.filter({expr})",
        f"{len(filtered)} rows matched",
    )


# ---------------------------------------------------------------------------
# Operation 2: Select columns (projection)
# ---------------------------------------------------------------------------


def demo_select(rows: list[dict[str, Any]]) -> None:
    """Show projection pattern: pandas column selection → Polars select."""
    _header("2. Select / project columns")

    keep = ["order_id", "customer_id", "amount"]
    projected = [{k: r[k] for k in keep} for r in rows[:3]]

    _show(
        "Select",
        "df[['order_id', 'customer_id', 'amount']]",
        'df.select(["order_id", "customer_id", "amount"])',
        f"first 3 rows: {projected}",
    )


# ---------------------------------------------------------------------------
# Operation 3: Add / derive a column
# ---------------------------------------------------------------------------


def demo_with_columns(rows: list[dict[str, Any]]) -> None:
    """Show derived column pattern: pandas assign → Polars with_columns."""
    _header("3. Derive new column (revenue = amount * quantity)")

    enriched = [{**r, "revenue": round(r["amount"] * r["quantity"], 2)} for r in rows[:3]]

    expr = (
        ExpressionBuilder("amount")
        .alias("_")  # demonstrates the builder
        .build()
    )
    polars_expr = 'pl.col("amount") * pl.col("quantity").alias("revenue")'

    _show(
        "WithColumns",
        "df.assign(revenue=df['amount'] * df['quantity'])",
        f"df.with_columns({polars_expr})  # ExpressionBuilder: {expr}",
        f"first 3 rows revenue: {[r['revenue'] for r in enriched]}",
    )


# ---------------------------------------------------------------------------
# Operation 4: GroupBy + aggregation
# ---------------------------------------------------------------------------


def demo_groupby(rows: list[dict[str, Any]]) -> None:
    """Show groupby pattern: pandas groupby → Polars group_by + agg."""
    _header("4. GroupBy + aggregation (total amount per category)")

    # Pure-Python groupby
    totals: dict[str, float] = defaultdict(float)
    for r in rows:
        totals[r["category"]] += r["amount"]
    totals_sorted = sorted(totals.items(), key=lambda x: x[1], reverse=True)

    # Polars expression string
    expr = build_aggregation(
        group_by=["category"],
        aggs={"amount": AggFunction.SUM, "order_id": AggFunction.COUNT},
        aliases={"amount": "total_amount", "order_id": "order_count"},
    )

    _show(
        "GroupBy",
        "df.groupby('category')['amount'].sum()",
        expr,
        f"top category: {totals_sorted[0]}",
    )


# ---------------------------------------------------------------------------
# Operation 5: Sort
# ---------------------------------------------------------------------------


def demo_sort(rows: list[dict[str, Any]]) -> None:
    """Show sort pattern: pandas sort_values → Polars sort."""
    _header("5. Sort by amount descending")

    top5 = sorted(rows, key=lambda r: r["amount"], reverse=True)[:5]

    _show(
        "Sort",
        "df.sort_values('amount', ascending=False).head(5)",
        'df.sort("amount", descending=True).head(5)',
        f"top 5 amounts: {[r['amount'] for r in top5]}",
    )


# ---------------------------------------------------------------------------
# Operation 6: Join / merge
# ---------------------------------------------------------------------------


def demo_join(rows: list[dict[str, Any]]) -> None:
    """Show join pattern: pandas merge → Polars join."""
    _header("6. Join two tables (sales LEFT JOIN product_info)")

    # Build a simple product_info lookup
    product_info = {
        f"prod_{i:03d}": {"product": f"prod_{i:03d}", "brand": f"brand_{(i % 5) + 1}"}
        for i in range(1, 21)
    }

    # Pure-Python left-join
    joined = [
        {**r, "brand": product_info.get(r["product"], {}).get("brand", None)} for r in rows[:3]
    ]

    _show(
        "Join",
        "df.merge(product_info_df, on='product', how='left')",
        'df.join(product_info_lf, on="product", how="left")',
        f"first 3 joined rows brands: {[r['brand'] for r in joined]}",
    )


# ---------------------------------------------------------------------------
# Operation 7: Window function (rolling sum)
# ---------------------------------------------------------------------------


def demo_window(rows: list[dict[str, Any]]) -> None:
    """Show window function pattern: pandas rolling → Polars .over()."""
    _header("7. Window function: rolling sum of amount over customer partitions")

    # Group rows by customer_id, compute rolling sum within each group
    customer_amounts: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        customer_amounts[r["customer_id"]].append(r["amount"])

    # Rolling sum (window=3) for the customer with most orders
    busiest_cid = max(customer_amounts, key=lambda k: len(customer_amounts[k]))
    cust1 = customer_amounts[busiest_cid]
    rolling = [round(sum(cust1[max(0, i - 2) : i + 1]), 2) for i in range(min(5, len(cust1)))]

    spec = WindowSpec(partition_by=["customer_id"], order_by="order_id", window_size=3)
    expr = build_window_expr("amount", WindowFunction.ROLLING_SUM, spec)

    _show(
        "Window",
        "df.groupby('customer_id')['amount'].transform(lambda x: x.rolling(3).sum())",
        f"df.with_columns({expr})",
        f"rolling(3) for customer_id=1 (first 5): {rolling}",
    )


# ---------------------------------------------------------------------------
# Operation 8: Null handling
# ---------------------------------------------------------------------------


def demo_null_handling(rows: list[dict[str, Any]]) -> None:
    """Show null-handling pattern: pandas fillna/dropna → Polars fill_null/drop_nulls."""
    _header("8. Null handling (fill_null / drop_nulls)")

    # Simulate nulls by replacing some amounts with None
    rng = random.Random(_RANDOM_SEED + 1)
    nullified = [{**r, "amount": None if rng.random() < 0.1 else r["amount"]} for r in rows[:20]]
    null_count = sum(1 for r in nullified if r["amount"] is None)
    filled = [{**r, "amount": 0.0 if r["amount"] is None else r["amount"]} for r in nullified]
    after_fill = sum(1 for r in filled if r["amount"] is None)

    expr_fill = ExpressionBuilder("amount").fill_null(0.0).build()
    expr_cast = (
        ExpressionBuilder("amount").cast("Float64").fill_null(0.0).alias("amount_clean").build()
    )

    _show(
        "FillNull",
        "df['amount'].fillna(0.0)",
        f"df.with_columns({expr_fill})  # chain example: {expr_cast}",
        f"nulls before={null_count}, after fill_null(0.0)={after_fill}",
    )


# ---------------------------------------------------------------------------
# Operation 9: Schema migration audit
# ---------------------------------------------------------------------------


def demo_schema_migration() -> None:
    """Show SchemaMapper + DataFrameConverter audit pattern."""
    _header("9. Schema migration: pandas dtypes → Polars schema")

    pandas_schema = {
        "order_id": "int64",
        "customer_id": "int32",
        "amount": "float64",
        "category": "object",
        "created_at": "datetime64[ns]",
        "is_active": "bool",
    }

    mapper = SchemaMapper(pandas_dtypes=pandas_schema)
    polars_schema = mapper.to_polars_schema()
    has_objects = mapper.has_object_columns()

    config = ConversionConfig(coerce_timestamps=True, rechunk=True, nan_to_null=True)
    converter = DataFrameConverter(config)

    df_info: dict[str, Any] = {
        "has_named_index": True,
        "uses_apply": True,
        "uses_inplace": False,
        "dtypes": pandas_schema,
    }
    issues = converter.audit_dataframe(df_info)

    print("\n  pandas schema → polars schema:")
    for col, dtype in polars_schema.items():
        print(f"    {col:15s} : {dtype}")

    print(f"\n  has_object_columns={has_objects}")
    print(f"  migration issues ({len(issues)} total):")
    for warning, msg in issues[:3]:
        print(f"    [{warning.value}] {msg}")


# ---------------------------------------------------------------------------
# Operation 10: Lazy pipeline expression builder
# ---------------------------------------------------------------------------


def demo_lazy_pipeline() -> None:
    """Show lazy pipeline builder: QueryPlan + build_lazy_pipeline."""
    _header("10. Lazy pipeline: scan → filter → group_by → collect")

    plan = QueryPlan(source="sales.parquet", scan_type=ScanType.PARQUET)
    plan.add_filter("amount", ">", 50.0)
    plan.add_filter("year", "==", 2024)
    plan.add_select(["customer_id", "category", "amount"])
    plan.add_group_by(
        ["category"],
        'pl.col("amount").sum().alias("total"), pl.col("customer_id").n_unique().alias("customers")',
    )
    plan.add_sort("total", descending=True)
    plan.add_limit(10)

    config = LazyPipelineConfig(
        predicate_pushdown=True,
        projection_pushdown=True,
        streaming=False,
    )
    pipeline_str = build_lazy_pipeline(plan, config)

    print("\n  Generated lazy pipeline expression:")
    # Pretty-print each chained call on its own line
    indent = "    "
    formatted = pipeline_str.replace(").", f")\n{indent}.")
    print(f"{indent}{formatted}")

    # Show optimizer suggestions
    from patterns.lazy_evaluation import PipelineOptimizer

    optimizer = PipelineOptimizer(plan, config)
    hints = optimizer.suggestions()
    print(f"\n  Optimizer hints ({len(hints)}):")
    for h in hints:
        print(f"    • {h}")


# ---------------------------------------------------------------------------
# Operation 11: Performance advisor
# ---------------------------------------------------------------------------


def demo_performance_advisor() -> None:
    """Show PerformanceAdvisor + profile_memory."""
    _header("11. Performance advisor: schema analysis + memory profiling")

    schema = {
        "order_id": "Int64",
        "customer_id": "Int32",
        "product": "Utf8",
        "category": "Categorical",
        "region": "Categorical",
        "amount": "Float64",
        "quantity": "Int32",
        "year": "Int32",
        "month": "Int8",
    }

    advisor = PerformanceAdvisor(
        schema=schema,
        usage={"uses_apply": False, "uses_eager": True, "sort_before_filter": True},
    )
    advice = advisor.advise()
    dtype_summary = advisor.dtype_summary()

    mem = profile_memory(schema, n_rows=1_000_000)

    print(f"\n  Schema dtype summary: {dtype_summary}")
    print(f"  Estimated memory for 1M rows: {mem.total_mb:.1f} MB")
    print(f"  Needs streaming (8 GB RAM limit): {mem.needs_streaming(8.0)}")
    print(f"\n  Performance advice ({len(advice)} hints):")
    for hint in advice:
        print(f"    • {hint}")

    # Show Timer usage
    with Timer("synthetic_sort", rows=200, operation="sort") as t:
        rows = make_sales_rows(200)
        rows.sort(key=lambda r: r["amount"])
    result = t.result
    assert result is not None
    print(
        f"\n  Timer demo — sort 200 rows: {result.ms:.3f} ms ({result.rows_per_second:,.0f} rows/s)"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all migration demonstrations."""
    print("\nPolars Migration Cookbook — Migration Demo")
    print("==========================================")
    print("All data is synthetic (stdlib only). Expression strings show real")
    print("Polars code; Python stdlib executes equivalent logic for verification.\n")

    rows = make_sales_rows(200)

    demo_filter(rows)
    demo_select(rows)
    demo_with_columns(rows)
    demo_groupby(rows)
    demo_sort(rows)
    demo_join(rows)
    demo_window(rows)
    demo_null_handling(rows)
    demo_schema_migration()
    demo_lazy_pipeline()
    demo_performance_advisor()

    print("\n\nAll demos completed successfully.")


if __name__ == "__main__":
    main()
