# polars-migration-cookbook

Reusable patterns for migrating pandas codebases to Polars: lazy evaluation,
expression builders, performance profiling, and pandas compatibility helpers.

## Patterns

- **expressions** — filter/aggregation/window expression builders, fluent `ExpressionBuilder`
- **lazy_evaluation** — `QueryPlan`, `PipelineOptimizer`, `LazyPipelineConfig`, `build_lazy_pipeline()`
- **pandas_compat** — dtype mapping, `SchemaMapper`, `DataFrameConverter` migration audit
- **performance** — `BenchmarkResult`, `MemoryProfile`, `PerformanceAdvisor`, `Timer`

## Install

```bash
pip install -e ".[dev]"
pytest
```
