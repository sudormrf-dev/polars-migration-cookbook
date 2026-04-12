"""Polars migration cookbook: pandas → Polars patterns."""

from .expressions import (
    ExpressionBuilder,
    WindowSpec,
    build_aggregation,
    build_filter,
    build_window_expr,
)
from .lazy_evaluation import (
    LazyPipelineConfig,
    LazyPipelineStats,
    PipelineOptimizer,
    QueryPlan,
    build_lazy_pipeline,
)
from .pandas_compat import (
    ConversionConfig,
    DataFrameConverter,
    MigrationWarning,
    SchemaMapper,
    infer_polars_dtype,
)
from .performance import (
    BenchmarkResult,
    MemoryProfile,
    ParallelConfig,
    PerformanceAdvisor,
    profile_memory,
)

__all__ = [
    "BenchmarkResult",
    "ConversionConfig",
    "DataFrameConverter",
    "ExpressionBuilder",
    "LazyPipelineConfig",
    "LazyPipelineStats",
    "MemoryProfile",
    "MigrationWarning",
    "ParallelConfig",
    "PerformanceAdvisor",
    "PipelineOptimizer",
    "QueryPlan",
    "SchemaMapper",
    "WindowSpec",
    "build_aggregation",
    "build_filter",
    "build_lazy_pipeline",
    "build_window_expr",
    "infer_polars_dtype",
    "profile_memory",
]
