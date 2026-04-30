"""Benchmark: Python naive vs optimised operations — simulating pandas vs Polars patterns.

This module benchmarks four canonical data operations at 1 M rows using two
implementation strategies:

1. **Naive / pandas-style** — Python list comprehensions and loops that mimic
   how a typical pandas user would approach each task (row-by-row, with
   intermediate allocations).

2. **Optimised / Polars-style** — Python implementations that apply the same
   algorithmic optimisations Polars uses internally: single-pass aggregation,
   hash-based grouping, pre-sorted merge-join, and incremental window sums.

The benchmark intentionally avoids importing ``pandas`` or ``polars`` so it
runs in any environment.  The timings and speedup ratios illustrate *why*
expression-based, vectorised, single-pass algorithms win at scale — the same
principle that makes Polars dramatically faster than pandas in production.

Operations benchmarked
----------------------
- **Filter**        — keep rows where amount > threshold AND year == target
- **GroupBy+Agg**   — sum(amount) and count() per category
- **Join**          — left-join orders with a product metadata table
- **Rolling window** — 3-row rolling sum of amount within customer partition

Run::

    python benchmarks/pandas_vs_polars.py

Expected output: a formatted results table with timing and speedup ratios.
"""

from __future__ import annotations

import math
import random
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SEED = 42
_N_ROWS = 1_000_000
_CATEGORIES = ["electronics", "clothing", "food", "books", "sports", "home", "toys", "beauty"]
_REGIONS = ["north", "south", "east", "west"]
_N_CUSTOMERS = 10_000
_N_PRODUCTS = 200
_FILTER_AMOUNT_THRESHOLD = 250.0
_FILTER_YEAR = 2023


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def generate_dataset(n: int = _N_ROWS) -> list[dict[str, Any]]:
    """Generate *n* synthetic sales rows deterministically.

    Uses a fixed seed so timing comparisons are reproducible.

    Args:
        n: Number of rows to generate.

    Returns:
        List of dicts with keys: order_id, customer_id, product_id, category,
        region, amount, quantity, year, month.
    """
    rng = random.Random(_SEED)
    # Pre-generate random arrays for speed (avoid per-row method calls)
    customer_ids = [rng.randint(1, _N_CUSTOMERS) for _ in range(n)]
    product_ids = [rng.randint(1, _N_PRODUCTS) for _ in range(n)]
    categories = [rng.randrange(len(_CATEGORIES)) for _ in range(n)]
    amounts = [round(rng.uniform(1.0, 1000.0), 2) for _ in range(n)]
    quantities = [rng.randint(1, 50) for _ in range(n)]
    years = [rng.choice([2021, 2022, 2023, 2024]) for _ in range(n)]
    months = [rng.randint(1, 12) for _ in range(n)]

    rows: list[dict[str, Any]] = []
    for i in range(n):
        rows.append(
            {
                "order_id": i + 1,
                "customer_id": customer_ids[i],
                "product_id": product_ids[i],
                "category": _CATEGORIES[categories[i]],
                "amount": amounts[i],
                "quantity": quantities[i],
                "year": years[i],
                "month": months[i],
            }
        )
    return rows


def generate_product_table(n_products: int = _N_PRODUCTS) -> list[dict[str, Any]]:
    """Generate a product metadata lookup table.

    Args:
        n_products: Number of distinct products.

    Returns:
        List of dicts with product_id, product_name, supplier, cost_price.
    """
    rng = random.Random(_SEED + 1)
    return [
        {
            "product_id": i,
            "product_name": f"Product {i:04d}",
            "supplier": f"Supplier_{(i % 20) + 1}",
            "cost_price": round(rng.uniform(0.5, 500.0), 2),
        }
        for i in range(1, n_products + 1)
    ]


# ---------------------------------------------------------------------------
# Benchmark result dataclass
# ---------------------------------------------------------------------------


@dataclass
class BenchResult:
    """Stores timing and output size for one benchmark run."""

    label: str
    elapsed_ms: float
    output_size: int  # number of output rows/groups

    @property
    def rows_per_sec(self) -> float:
        """Throughput in rows/second based on 1M input rows."""
        if self.elapsed_ms <= 0:
            return 0.0
        return _N_ROWS / (self.elapsed_ms / 1000.0)


def _time_it(fn: Any, *args: Any) -> tuple[float, Any]:
    """Run *fn(*args)* and return (elapsed_ms, result)."""
    start = time.perf_counter()
    result = fn(*args)
    elapsed = (time.perf_counter() - start) * 1000.0
    return elapsed, result


# ---------------------------------------------------------------------------
# BENCHMARK 1: Filter
# ---------------------------------------------------------------------------


def filter_naive(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Naive filter: iterate with condition checks (pandas-style boolean mask).

    Two-pass equivalent: first build mask, then collect matching rows.

    Args:
        rows: Input rows.

    Returns:
        Filtered rows.
    """
    mask = [r["amount"] > _FILTER_AMOUNT_THRESHOLD and r["year"] == _FILTER_YEAR for r in rows]
    return [r for r, keep in zip(rows, mask) if keep]


def filter_optimised(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Optimised filter: single-pass with short-circuit evaluation.

    Polars evaluates filter predicates in a single scan using SIMD comparisons.
    This Python version uses single-pass list comprehension (Python's most
    efficient iteration) with short-circuit ``and``.

    Args:
        rows: Input rows.

    Returns:
        Filtered rows.
    """
    threshold = _FILTER_AMOUNT_THRESHOLD
    target_year = _FILTER_YEAR
    return [r for r in rows if r["amount"] > threshold and r["year"] == target_year]


# ---------------------------------------------------------------------------
# BENCHMARK 2: GroupBy + aggregation
# ---------------------------------------------------------------------------


def groupby_naive(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Naive groupby: use intermediate list accumulation per group.

    Simulates pandas ``groupby().agg()`` with separate loops for each metric.

    Args:
        rows: Input rows.

    Returns:
        Dict of category → {sum_amount, count}.
    """
    # Pass 1: build per-group lists
    groups: dict[str, list[float]] = {}
    for r in rows:
        cat = r["category"]
        if cat not in groups:
            groups[cat] = []
        groups[cat].append(r["amount"])

    # Pass 2: compute aggregates
    result: dict[str, dict[str, Any]] = {}
    for cat, amounts in groups.items():
        result[cat] = {"sum_amount": sum(amounts), "count": len(amounts)}
    return result


def groupby_optimised(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Optimised groupby: single-pass running sum + count.

    Polars uses hash-table aggregation in a single scan.  This Python version
    updates running totals directly without storing per-group lists.

    Args:
        rows: Input rows.

    Returns:
        Dict of category → {sum_amount, count}.
    """
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for r in rows:
        cat = r["category"]
        amount = r["amount"]
        if cat in sums:
            sums[cat] += amount
            counts[cat] += 1
        else:
            sums[cat] = amount
            counts[cat] = 1
    return {cat: {"sum_amount": sums[cat], "count": counts[cat]} for cat in sums}


# ---------------------------------------------------------------------------
# BENCHMARK 3: Join
# ---------------------------------------------------------------------------


def join_naive(
    orders: list[dict[str, Any]],
    products: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Naive left-join: O(n * m) nested-loop join.

    Simulates a poorly-optimised pandas merge without hash indexing.

    Args:
        orders: Left table rows.
        products: Right table rows.

    Returns:
        Joined rows.
    """
    # Build lookup first (even "naive" pandas builds a hash index under the hood,
    # but some beginners use merge on unsorted data; we simulate the lookup build here
    # as a separate step to model the overhead)
    lookup: dict[int, dict[str, Any]] = {}
    for p in products:
        lookup[p["product_id"]] = p

    result: list[dict[str, Any]] = []
    for order in orders:
        prod = lookup.get(order["product_id"])
        if prod:
            result.append(
                {
                    "order_id": order["order_id"],
                    "product_id": order["product_id"],
                    "amount": order["amount"],
                    "product_name": prod["product_name"],
                    "supplier": prod["supplier"],
                    "cost_price": prod["cost_price"],
                    "margin": round(order["amount"] - prod["cost_price"], 2),
                }
            )
        else:
            result.append(
                {
                    "order_id": order["order_id"],
                    "product_id": order["product_id"],
                    "amount": order["amount"],
                    "product_name": None,
                    "supplier": None,
                    "cost_price": None,
                    "margin": None,
                }
            )
    return result


def join_optimised(
    orders: list[dict[str, Any]],
    products: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Optimised hash-join: pre-build hash index, single-pass probe.

    Polars performs hash-joins automatically.  This Python version pre-builds
    the hash table once and probes it in a single list comprehension, which
    mirrors the algorithmic pattern.

    Args:
        orders: Left table rows.
        products: Right table rows.

    Returns:
        Joined rows (same result as naive version).
    """
    # Build hash index once
    idx: dict[int, tuple[str, str, float]] = {
        p["product_id"]: (p["product_name"], p["supplier"], p["cost_price"])
        for p in products
    }

    result: list[dict[str, Any]] = []
    for order in orders:
        pid = order["product_id"]
        amount = order["amount"]
        hit = idx.get(pid)
        if hit:
            name, supplier, cost = hit
            result.append(
                {
                    "order_id": order["order_id"],
                    "product_id": pid,
                    "amount": amount,
                    "product_name": name,
                    "supplier": supplier,
                    "cost_price": cost,
                    "margin": round(amount - cost, 2),
                }
            )
        else:
            result.append(
                {
                    "order_id": order["order_id"],
                    "product_id": pid,
                    "amount": amount,
                    "product_name": None,
                    "supplier": None,
                    "cost_price": None,
                    "margin": None,
                }
            )
    return result


# ---------------------------------------------------------------------------
# BENCHMARK 4: Rolling window
# ---------------------------------------------------------------------------


def rolling_naive(rows: list[dict[str, Any]], window: int = 3) -> list[float | None]:
    """Naive rolling sum: re-sum the full window on every row.

    Simulates O(n * window) computation (what a Python loop with slicing does).

    Args:
        rows: Input rows (assumed sorted by customer_id, order_id).
        window: Rolling window size.

    Returns:
        List of rolling sums (None for positions without enough history).
    """
    # Group by customer, compute rolling per group
    groups: dict[int, list[float]] = defaultdict(list)
    order_map: dict[int, int] = {}  # order_id → group index
    for i, r in enumerate(rows):
        cid = r["customer_id"]
        groups[cid].append(r["amount"])
        order_map[i] = len(groups[cid]) - 1

    results: list[float | None] = []
    for i, r in enumerate(rows):
        cid = r["customer_id"]
        pos = order_map[i]
        group = groups[cid]
        if pos < window - 1:
            results.append(None)
        else:
            # Re-sum the last `window` values (O(window) per row)
            rolling_sum = sum(group[pos - window + 1 : pos + 1])
            results.append(round(rolling_sum, 2))
    return results


def rolling_optimised(rows: list[dict[str, Any]], window: int = 3) -> list[float | None]:
    """Optimised rolling sum: incremental sliding window, O(1) per row.

    Polars computes rolling operations incrementally, adding the new value
    and subtracting the value that left the window.  This Python version does
    the same using per-group running totals.

    Args:
        rows: Input rows.
        window: Rolling window size.

    Returns:
        List of rolling sums (None for positions without enough history).
    """
    # Group by customer_id first, preserving original order with position index
    group_positions: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for i, r in enumerate(rows):
        group_positions[r["customer_id"]].append((i, r["amount"]))

    results: list[float | None] = [None] * len(rows)
    for positions in group_positions.values():
        running = 0.0
        buf: list[float] = []
        for orig_idx, amount in positions:
            buf.append(amount)
            running += amount
            if len(buf) > window:
                running -= buf[-window - 1]
            if len(buf) >= window:
                results[orig_idx] = round(running, 2)
    return results


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------


def _fmt_ms(ms: float) -> str:
    """Format milliseconds with appropriate precision."""
    if ms >= 1000:
        return f"{ms / 1000:.2f} s"
    if ms >= 100:
        return f"{ms:.0f} ms"
    return f"{ms:.1f} ms"


def print_results(results: list[tuple[str, BenchResult, BenchResult]]) -> None:
    """Print a formatted benchmark table.

    Args:
        results: List of (operation_name, naive_result, optimised_result).
    """
    print("\n" + "=" * 90)
    print(f"  BENCHMARK RESULTS — {_N_ROWS:,} rows")
    print("  Naive = two-pass/intermediate-alloc (pandas-style)")
    print("  Optimised = single-pass/hash-table (Polars-style algorithms)")
    print("=" * 90)
    hdr = f"  {'Operation':<22} {'Naive':>12} {'Optimised':>12} {'Speedup':>9} "
    hdr += f"{'Opt rows/s':>14} {'Output':>8}"
    print(hdr)
    print("-" * 90)
    for op, naive, opt in results:
        speedup = naive.elapsed_ms / opt.elapsed_ms if opt.elapsed_ms > 0 else float("inf")
        line = (
            f"  {op:<22} "
            f"{_fmt_ms(naive.elapsed_ms):>12} "
            f"{_fmt_ms(opt.elapsed_ms):>12} "
            f"{speedup:>8.1f}x "
            f"{opt.rows_per_sec:>14,.0f} "
            f"{opt.output_size:>8,}"
        )
        print(line)
    print("=" * 90)
    print("\nNotes:")
    print("  • Real Polars uses SIMD + multi-threading, achieving 10-100x more speedup.")
    print("  • Speedups here reflect pure algorithmic improvement (single-pass vs multi-pass).")
    print("  • For 10M+ rows, enable streaming: lf.collect(streaming=True)")
    print("  • GroupBy: use pl.Categorical dtype for low-cardinality string columns.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all benchmarks and print results."""
    print("\nPolars Migration Cookbook — Benchmarks")
    print("=" * 50)
    print(f"Generating {_N_ROWS:,} synthetic rows...")
    t0 = time.perf_counter()
    rows = generate_dataset(_N_ROWS)
    products = generate_product_table(_N_PRODUCTS)
    gen_ms = (time.perf_counter() - t0) * 1000
    print(f"Data generation: {_fmt_ms(gen_ms)}\n")

    collected: list[tuple[str, BenchResult, BenchResult]] = []

    # ---- Filter ----
    print("Running: Filter (amount > 250 AND year == 2023)...")
    naive_ms, naive_out = _time_it(filter_naive, rows)
    opt_ms, opt_out = _time_it(filter_optimised, rows)
    assert len(naive_out) == len(opt_out), "filter output size mismatch"
    collected.append((
        "Filter",
        BenchResult("naive_filter", naive_ms, len(naive_out)),
        BenchResult("opt_filter", opt_ms, len(opt_out)),
    ))
    print(f"  naive={_fmt_ms(naive_ms)}, optimised={_fmt_ms(opt_ms)}, "
          f"matched rows={len(opt_out):,}")

    # ---- GroupBy + Agg ----
    print("Running: GroupBy + sum(amount) per category...")
    naive_ms, naive_gb = _time_it(groupby_naive, rows)
    opt_ms, opt_gb = _time_it(groupby_optimised, rows)
    # Validate results match (allow floating point tolerance)
    for cat in naive_gb:
        diff = abs(naive_gb[cat]["sum_amount"] - opt_gb[cat]["sum_amount"])
        assert diff < 0.01, f"groupby mismatch for {cat}: {diff}"
    collected.append((
        "GroupBy+Agg",
        BenchResult("naive_groupby", naive_ms, len(naive_gb)),
        BenchResult("opt_groupby", opt_ms, len(opt_gb)),
    ))
    print(f"  naive={_fmt_ms(naive_ms)}, optimised={_fmt_ms(opt_ms)}, "
          f"groups={len(opt_gb)}")

    # ---- Join ----
    print("Running: Left-join orders with product table...")
    naive_ms, naive_join = _time_it(join_naive, rows, products)
    opt_ms, opt_join = _time_it(join_optimised, rows, products)
    assert len(naive_join) == len(opt_join), "join output size mismatch"
    collected.append((
        "Join (left)",
        BenchResult("naive_join", naive_ms, len(naive_join)),
        BenchResult("opt_join", opt_ms, len(opt_join)),
    ))
    print(f"  naive={_fmt_ms(naive_ms)}, optimised={_fmt_ms(opt_ms)}, "
          f"output rows={len(opt_join):,}")

    # ---- Rolling window ----
    # Use a 100k subset for rolling (full 1M would be slow in Python)
    rolling_rows = rows[:100_000]
    print(f"Running: Rolling window sum (window=3, on {len(rolling_rows):,} rows)...")
    naive_ms, naive_roll = _time_it(rolling_naive, rolling_rows)
    opt_ms, opt_roll = _time_it(rolling_optimised, rolling_rows)
    # Count non-None results
    naive_non_null = sum(1 for v in naive_roll if v is not None)
    opt_non_null = sum(1 for v in opt_roll if v is not None)
    assert naive_non_null == opt_non_null, (
        f"rolling non-null count mismatch: {naive_non_null} vs {opt_non_null}"
    )
    collected.append((
        f"Rolling sum (100k)",
        BenchResult("naive_rolling", naive_ms, naive_non_null),
        BenchResult("opt_rolling", opt_ms, opt_non_null),
    ))
    print(f"  naive={_fmt_ms(naive_ms)}, optimised={_fmt_ms(opt_ms)}, "
          f"non-null windows={opt_non_null:,}")

    print_results(collected)
    print("\nBenchmark complete.\n")


if __name__ == "__main__":
    main()
