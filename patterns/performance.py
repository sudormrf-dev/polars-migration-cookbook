"""Polars performance patterns: benchmarking, memory profiling, parallelism.

Polars is multi-threaded by default and uses Apache Arrow memory layout.
Key performance levers:
  - Avoid .apply()/.map_elements() — use native expressions
  - Use lazy API + collect() for query optimization
  - Rechunk after concat/vertical_stack to defragment memory
  - Use Categorical dtype for low-cardinality string columns
  - Use streaming for datasets > RAM

Patterns:
  - BenchmarkResult: captures timing and throughput for a query
  - MemoryProfile: tracks estimated Arrow memory usage
  - ParallelConfig: configure thread pool and SIMD settings
  - PerformanceAdvisor: analyses schema/usage patterns for advice
  - profile_memory(): estimate memory for a given schema + row count

Usage::

    config = ParallelConfig(n_threads=8, use_streaming=True)
    advisor = PerformanceAdvisor(schema={"name": "Utf8", "age": "Int32"})
    advice = advisor.advise()
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class BottleneckType(str, Enum):
    """Category of performance bottleneck."""

    APPLY_USAGE = "apply_usage"
    HIGH_CARDINALITY_STRING = "high_cardinality_string"
    FRAGMENTED_MEMORY = "fragmented_memory"
    MISSING_LAZY = "missing_lazy"
    SORT_BEFORE_FILTER = "sort_before_filter"
    REPEATED_SCAN = "repeated_scan"
    OBJECT_DTYPE = "object_dtype"


# Approximate bytes per value for each Polars dtype
_DTYPE_BYTES: dict[str, int] = {
    "Boolean": 1,
    "Int8": 1,
    "Int16": 2,
    "Int32": 4,
    "Int64": 8,
    "UInt8": 1,
    "UInt16": 2,
    "UInt32": 4,
    "UInt64": 8,
    "Float32": 4,
    "Float64": 8,
    "Date": 4,
    "Datetime": 8,
    "Duration": 8,
    "Utf8": 32,  # estimate: avg 32 bytes per string value
    "String": 32,
    "Categorical": 4,  # dictionary-encoded
    "List": 24,  # rough estimate
    "Struct": 16,  # rough estimate
    "Unknown": 8,
}


def profile_memory(schema: dict[str, str], n_rows: int) -> MemoryProfile:
    """Estimate Arrow memory usage for a schema and row count.

    Args:
        schema: Dict of column_name → Polars dtype string.
        n_rows: Number of rows.

    Returns:
        :class:`MemoryProfile` with per-column and total estimates.
    """
    per_column: dict[str, int] = {}
    for col, dtype in schema.items():
        bytes_per = _DTYPE_BYTES.get(dtype, 8)
        per_column[col] = bytes_per * n_rows

    total = sum(per_column.values())
    return MemoryProfile(
        schema=schema, n_rows=n_rows, per_column_bytes=per_column, total_bytes=total
    )


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run.

    Attributes:
        label: Human-readable label for this benchmark.
        elapsed_seconds: Wall-clock time in seconds.
        rows_processed: Number of rows processed.
        operation: Description of the operation benchmarked.
    """

    label: str
    elapsed_seconds: float
    rows_processed: int = 0
    operation: str = ""

    @property
    def rows_per_second(self) -> float:
        """Throughput in rows/second."""
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.rows_processed / self.elapsed_seconds

    @property
    def ms(self) -> float:
        """Elapsed time in milliseconds."""
        return self.elapsed_seconds * 1000.0

    def is_faster_than(self, other: BenchmarkResult) -> bool:
        """Return True if this result is faster than *other*."""
        return self.elapsed_seconds < other.elapsed_seconds

    def speedup_vs(self, other: BenchmarkResult) -> float:
        """Return speedup ratio relative to *other* (other.elapsed / self.elapsed).

        Returns 1.0 if either elapsed is zero.
        """
        if self.elapsed_seconds <= 0 or other.elapsed_seconds <= 0:
            return 1.0
        return other.elapsed_seconds / self.elapsed_seconds


@dataclass
class MemoryProfile:
    """Estimated Arrow memory usage for a DataFrame schema.

    Attributes:
        schema: Column → dtype mapping.
        n_rows: Row count used for the estimate.
        per_column_bytes: Estimated bytes per column.
        total_bytes: Total estimated bytes.
    """

    schema: dict[str, str] = field(default_factory=dict)
    n_rows: int = 0
    per_column_bytes: dict[str, int] = field(default_factory=dict)
    total_bytes: int = 0

    @property
    def total_mb(self) -> float:
        """Total estimated memory in megabytes."""
        return self.total_bytes / (1024 * 1024)

    @property
    def total_gb(self) -> float:
        """Total estimated memory in gigabytes."""
        return self.total_bytes / (1024 * 1024 * 1024)

    def largest_columns(self, top_n: int = 3) -> list[tuple[str, int]]:
        """Return the top N columns by estimated byte size."""
        sorted_cols = sorted(self.per_column_bytes.items(), key=lambda x: x[1], reverse=True)
        return sorted_cols[:top_n]

    def needs_streaming(self, available_ram_gb: float = 8.0) -> bool:
        """Return True if the estimated size exceeds available RAM."""
        return self.total_gb > available_ram_gb


@dataclass
class ParallelConfig:
    """Configuration for Polars parallel execution.

    Attributes:
        n_threads: Number of threads for the Polars thread pool (0 = auto).
        use_streaming: Enable streaming for out-of-core computation.
        chunk_size: Rows per chunk in streaming mode.
        string_cache: Enable global string cache for Categorical joins.
        rechunk_after_concat: Rechunk after concat operations.
    """

    n_threads: int = 0
    use_streaming: bool = False
    chunk_size: int = 50_000
    string_cache: bool = False
    rechunk_after_concat: bool = True

    def env_vars(self) -> dict[str, str]:
        """Return environment variables to set for this config."""
        env: dict[str, str] = {}
        if self.n_threads > 0:
            env["POLARS_MAX_THREADS"] = str(self.n_threads)
        return env

    def is_streaming_recommended(self, memory_profile: MemoryProfile) -> bool:
        """Return True if streaming is recommended given the memory profile."""
        return memory_profile.total_gb > 4.0 or self.use_streaming


class PerformanceAdvisor:
    """Analyses a Polars schema and usage patterns for performance advice.

    Args:
        schema: Dict of column_name → Polars dtype string.
        usage: Optional dict describing usage patterns (see :meth:`advise`).
    """

    def __init__(
        self,
        schema: dict[str, str],
        usage: dict[str, Any] | None = None,
    ) -> None:
        self._schema = schema
        self._usage = usage or {}

    @property
    def schema(self) -> dict[str, str]:
        """The DataFrame schema being analysed."""
        return self._schema

    def detect_bottlenecks(self) -> list[tuple[BottleneckType, str]]:
        """Scan schema and usage for known bottlenecks.

        Usage dict keys:
            - ``uses_apply``: bool — .apply() detected
            - ``uses_eager``: bool — no .lazy() used
            - ``sort_before_filter``: bool — sort before filter
            - ``concat_count``: int — number of concat calls
            - ``scan_count``: int — number of scans of same file

        Returns:
            List of (bottleneck_type, message) tuples.
        """
        issues: list[tuple[BottleneckType, str]] = []

        if self._usage.get("uses_apply"):
            issues.append(
                (
                    BottleneckType.APPLY_USAGE,
                    "map_elements()/.apply() detected — replace with native Polars expressions",
                )
            )

        for col, dtype in self._schema.items():
            if dtype in {"Utf8", "String"}:
                issues.append(
                    (
                        BottleneckType.HIGH_CARDINALITY_STRING,
                        f"Column '{col}' is {dtype} — consider Categorical for low-cardinality strings",
                    )
                )

        if self._usage.get("uses_eager") and len(self._schema) > 5:
            issues.append(
                (
                    BottleneckType.MISSING_LAZY,
                    "Eager evaluation on wide schema — use .lazy().filter().collect() for optimization",
                )
            )

        if self._usage.get("sort_before_filter"):
            issues.append(
                (
                    BottleneckType.SORT_BEFORE_FILTER,
                    "Sort before filter — filter first to reduce rows, then sort",
                )
            )

        concat_count = int(self._usage.get("concat_count", 0))
        if concat_count > 3:
            issues.append(
                (
                    BottleneckType.FRAGMENTED_MEMORY,
                    f"{concat_count} concat calls — rechunk() after final concat to defragment",
                )
            )

        scan_count = int(self._usage.get("scan_count", 0))
        if scan_count > 1:
            issues.append(
                (
                    BottleneckType.REPEATED_SCAN,
                    f"File scanned {scan_count} times — cache with .collect() or use LazyFrame.cache()",
                )
            )

        for col, dtype in self._schema.items():
            if dtype.lower() == "object":
                issues.append(
                    (
                        BottleneckType.OBJECT_DTYPE,
                        f"Column '{col}' has object dtype — use typed Polars dtype instead",
                    )
                )

        return issues

    def advise(self) -> list[str]:
        """Return plain-language performance advice."""
        bottlenecks = self.detect_bottlenecks()
        return [msg for _, msg in bottlenecks]

    def dtype_summary(self) -> dict[str, int]:
        """Return count of each dtype in the schema."""
        counts: dict[str, int] = {}
        for dtype in self._schema.values():
            counts[dtype] = counts.get(dtype, 0) + 1
        return counts


class Timer:
    """Simple context manager for benchmarking code blocks.

    Usage::

        with Timer("my_query", rows=1_000_000) as t:
            # ... run query ...
        result = t.result
    """

    def __init__(self, label: str, rows: int = 0, operation: str = "") -> None:
        self.label = label
        self.rows = rows
        self.operation = operation
        self._start: float = 0.0
        self.result: BenchmarkResult | None = None

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: object) -> None:
        elapsed = time.perf_counter() - self._start
        self.result = BenchmarkResult(
            label=self.label,
            elapsed_seconds=elapsed,
            rows_processed=self.rows,
            operation=self.operation,
        )
