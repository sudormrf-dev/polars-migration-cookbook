"""Polars lazy evaluation patterns: query plans, pipeline optimization.

Polars lazy evaluation defers computation until .collect() is called,
enabling predicate pushdown, projection pushdown, and parallel execution.

Key patterns:
  - QueryPlan: represents a lazy query with scan/filter/select/group_by steps
  - LazyPipelineConfig: configuration for lazy pipeline execution
  - LazyPipelineStats: tracks stats from query plan inspection
  - PipelineOptimizer: analyzes and suggests optimizations
  - build_lazy_pipeline(): constructs lazy pipeline expression strings

Usage::

    config = LazyPipelineConfig(predicate_pushdown=True, streaming=False)
    plan = QueryPlan(source="orders.parquet")
    plan.add_filter("amount", ">", 100)
    plan.add_select(["order_id", "amount", "customer_id"])
    expr = build_lazy_pipeline(plan, config)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class OptimizationFlag(str, Enum):
    """Polars query optimizer flags."""

    PREDICATE_PUSHDOWN = "predicate_pushdown"
    PROJECTION_PUSHDOWN = "projection_pushdown"
    SIMPLIFY_EXPRESSION = "simplify_expression"
    SLICE_PUSHDOWN = "slice_pushdown"
    COMM_SUBPLAN_ELIM = "comm_subplan_elim"
    STREAMING = "streaming"


class ScanType(str, Enum):
    """Supported lazy scan sources."""

    CSV = "csv"
    PARQUET = "parquet"
    IPC = "ipc"
    NDJSON = "ndjson"
    DATABASE = "database"
    IN_MEMORY = "in_memory"


@dataclass
class QueryStep:
    """A single step in a lazy query plan.

    Attributes:
        operation: Step type (filter, select, group_by, join, sort, etc.).
        args: Positional arguments for the step.
        kwargs: Keyword arguments for the step.
    """

    operation: str
    args: list[Any] = field(default_factory=list)
    kwargs: dict[str, Any] = field(default_factory=dict)

    def to_expr(self) -> str:
        """Render the step as a chained method call string."""
        arg_parts: list[str] = [repr(a) for a in self.args]
        kwarg_parts: list[str] = [f"{k}={v!r}" for k, v in self.kwargs.items()]
        all_parts = arg_parts + kwarg_parts
        return f".{self.operation}({', '.join(all_parts)})"


@dataclass
class QueryPlan:
    """Represents a lazy Polars query plan.

    Attributes:
        source: File path or table name to scan.
        scan_type: How to read the source.
        steps: Ordered list of query steps.
    """

    source: str = ""
    scan_type: ScanType = ScanType.PARQUET
    steps: list[QueryStep] = field(default_factory=list)

    def add_filter(self, column: str, op: str, value: Any) -> QueryPlan:
        """Add a filter step."""
        self.steps.append(QueryStep("filter", args=[f'pl.col("{column}") {op} {value!r}']))
        return self

    def add_select(self, columns: list[str]) -> QueryPlan:
        """Add a projection step."""
        col_list = ", ".join(f'"{c}"' for c in columns)
        self.steps.append(QueryStep("select", args=[f"[{col_list}]"]))
        return self

    def add_group_by(self, by: list[str], agg_expr: str) -> QueryPlan:
        """Add a group_by + agg step."""
        by_list = ", ".join(f'"{c}"' for c in by)
        self.steps.append(QueryStep("group_by", args=[f"[{by_list}]"]))
        self.steps.append(QueryStep("agg", args=[agg_expr]))
        return self

    def add_sort(self, column: str, descending: bool = False) -> QueryPlan:
        """Add a sort step."""
        self.steps.append(QueryStep("sort", args=[column], kwargs={"descending": descending}))
        return self

    def add_limit(self, n: int) -> QueryPlan:
        """Add a limit/head step."""
        self.steps.append(QueryStep("limit", args=[n]))
        return self

    def add_with_columns(self, *exprs: str) -> QueryPlan:
        """Add a with_columns step for derived columns."""
        self.steps.append(QueryStep("with_columns", args=list(exprs)))
        return self

    def filter_steps(self) -> list[QueryStep]:
        """Return only filter steps."""
        return [s for s in self.steps if s.operation == "filter"]

    def select_steps(self) -> list[QueryStep]:
        """Return only select/projection steps."""
        return [s for s in self.steps if s.operation == "select"]

    def count_steps(self, operation: str) -> int:
        """Count how many steps match the given operation name."""
        return sum(1 for s in self.steps if s.operation == operation)


@dataclass
class LazyPipelineConfig:
    """Configuration for a lazy Polars pipeline.

    Attributes:
        predicate_pushdown: Push filters as close to the scan as possible.
        projection_pushdown: Read only the needed columns.
        simplify_expression: Simplify constant sub-expressions.
        slice_pushdown: Push slice operations down.
        streaming: Enable streaming mode for larger-than-RAM datasets.
        n_rows: Optional row limit applied at scan time.
        low_memory: Reduce memory at cost of speed.
    """

    predicate_pushdown: bool = True
    projection_pushdown: bool = True
    simplify_expression: bool = True
    slice_pushdown: bool = True
    streaming: bool = False
    n_rows: int | None = None
    low_memory: bool = False

    def active_optimizations(self) -> list[OptimizationFlag]:
        """Return the list of active optimization flags."""
        flags: list[OptimizationFlag] = []
        if self.predicate_pushdown:
            flags.append(OptimizationFlag.PREDICATE_PUSHDOWN)
        if self.projection_pushdown:
            flags.append(OptimizationFlag.PROJECTION_PUSHDOWN)
        if self.simplify_expression:
            flags.append(OptimizationFlag.SIMPLIFY_EXPRESSION)
        if self.slice_pushdown:
            flags.append(OptimizationFlag.SLICE_PUSHDOWN)
        if self.streaming:
            flags.append(OptimizationFlag.STREAMING)
        return flags

    def collect_kwargs(self) -> dict[str, Any]:
        """Return kwargs dict for .collect()/.sink_parquet()."""
        kwargs: dict[str, Any] = {
            "predicate_pushdown": self.predicate_pushdown,
            "projection_pushdown": self.projection_pushdown,
            "simplify_expression": self.simplify_expression,
            "slice_pushdown": self.slice_pushdown,
            "streaming": self.streaming,
        }
        if self.low_memory:
            kwargs["low_memory"] = True
        return kwargs


@dataclass
class LazyPipelineStats:
    """Statistics gathered from inspecting a lazy query plan.

    Attributes:
        filter_count: Number of filter operations.
        projection_count: Number of select/projection operations.
        join_count: Number of join operations.
        sort_count: Number of sort operations.
        group_by_count: Number of group_by operations.
        estimated_pushdown_savings: Estimated column reduction from pushdown.
    """

    filter_count: int = 0
    projection_count: int = 0
    join_count: int = 0
    sort_count: int = 0
    group_by_count: int = 0
    estimated_pushdown_savings: float = 0.0

    @property
    def total_steps(self) -> int:
        """Total number of tracked steps."""
        return (
            self.filter_count
            + self.projection_count
            + self.join_count
            + self.sort_count
            + self.group_by_count
        )

    def is_complex(self) -> bool:
        """Return True if the pipeline has joins or multiple group_bys."""
        return self.join_count > 0 or self.group_by_count > 1


class PipelineOptimizer:
    """Analyzes a QueryPlan and suggests optimizations.

    Args:
        plan: The query plan to analyze.
        config: Pipeline configuration.
    """

    def __init__(self, plan: QueryPlan, config: LazyPipelineConfig | None = None) -> None:
        self._plan = plan
        self._cfg = config or LazyPipelineConfig()

    @property
    def plan(self) -> QueryPlan:
        """The query plan being optimized."""
        return self._plan

    @property
    def config(self) -> LazyPipelineConfig:
        """The pipeline configuration."""
        return self._cfg

    def gather_stats(self) -> LazyPipelineStats:
        """Collect statistics about the query plan."""
        return LazyPipelineStats(
            filter_count=self._plan.count_steps("filter"),
            projection_count=self._plan.count_steps("select"),
            join_count=self._plan.count_steps("join"),
            sort_count=self._plan.count_steps("sort"),
            group_by_count=self._plan.count_steps("group_by"),
        )

    def suggestions(self) -> list[str]:
        """Return optimization suggestions for this pipeline."""
        hints: list[str] = []
        stats = self.gather_stats()

        if stats.filter_count > 0 and not self._cfg.predicate_pushdown:
            hints.append("Enable predicate_pushdown to move filters closer to scan")

        if stats.projection_count > 0 and not self._cfg.projection_pushdown:
            hints.append("Enable projection_pushdown to read only needed columns")

        if stats.sort_count > 1:
            hints.append("Multiple sorts detected — consolidate into a single sort step")

        if stats.join_count > 0 and not self._cfg.streaming:
            hints.append("Joins present — consider streaming=True for large datasets")

        if self._plan.scan_type == ScanType.CSV:
            hints.append("CSV scan detected — convert to Parquet for faster reads")

        if stats.group_by_count > 1:
            hints.append("Multiple group_bys — verify they cannot be merged into one")

        return hints

    def reorder_for_pushdown(self) -> list[QueryStep]:
        """Return steps reordered so filters come before other operations."""
        filters = [s for s in self._plan.steps if s.operation == "filter"]
        others = [s for s in self._plan.steps if s.operation != "filter"]
        return filters + others


def build_lazy_pipeline(plan: QueryPlan, config: LazyPipelineConfig | None = None) -> str:
    """Build a lazy Polars pipeline expression string.

    Args:
        plan: The query plan to render.
        config: Optional pipeline configuration.

    Returns:
        Python expression string for the lazy pipeline.
    """
    cfg = config or LazyPipelineConfig()

    # Determine scan function
    scan_fn_map: dict[ScanType, str] = {
        ScanType.PARQUET: "scan_parquet",
        ScanType.CSV: "scan_csv",
        ScanType.IPC: "scan_ipc",
        ScanType.NDJSON: "scan_ndjson",
        ScanType.IN_MEMORY: "lazy",
        ScanType.DATABASE: "read_database_uri",
    }
    scan_fn = scan_fn_map.get(plan.scan_type, "scan_parquet")

    if plan.scan_type == ScanType.IN_MEMORY:
        expr = f"df.{scan_fn}()"
    else:
        n_rows_part = f", n_rows={cfg.n_rows}" if cfg.n_rows is not None else ""
        expr = f'pl.{scan_fn}("{plan.source}"{n_rows_part})'

    for step in plan.steps:
        expr += step.to_expr()

    # Add .collect() with optimizer kwargs
    kw = cfg.collect_kwargs()
    kw_str = ", ".join(f"{k}={v!r}" for k, v in kw.items())
    expr += f".collect({kw_str})"

    return expr
