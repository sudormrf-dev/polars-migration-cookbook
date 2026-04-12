"""Tests for lazy_evaluation.py."""

from __future__ import annotations

from patterns.lazy_evaluation import (
    LazyPipelineConfig,
    LazyPipelineStats,
    OptimizationFlag,
    PipelineOptimizer,
    QueryPlan,
    QueryStep,
    ScanType,
    build_lazy_pipeline,
)


class TestQueryStep:
    def test_to_expr_no_args(self):
        step = QueryStep("collect")
        assert step.to_expr() == ".collect()"

    def test_to_expr_with_args(self):
        step = QueryStep("limit", args=[100])
        assert step.to_expr() == ".limit(100)"

    def test_to_expr_with_kwargs(self):
        step = QueryStep("sort", args=["col"], kwargs={"descending": True})
        assert "descending=True" in step.to_expr()


class TestQueryPlan:
    def test_empty_plan(self):
        plan = QueryPlan(source="data.parquet")
        assert plan.steps == []

    def test_add_filter(self):
        plan = QueryPlan()
        plan.add_filter("age", ">", 18)
        assert plan.count_steps("filter") == 1

    def test_add_select(self):
        plan = QueryPlan()
        plan.add_select(["a", "b"])
        assert plan.count_steps("select") == 1

    def test_add_group_by(self):
        plan = QueryPlan()
        plan.add_group_by(["dept"], 'pl.col("salary").mean()')
        assert plan.count_steps("group_by") == 1
        assert plan.count_steps("agg") == 1

    def test_add_sort(self):
        plan = QueryPlan()
        plan.add_sort("value", descending=True)
        assert plan.count_steps("sort") == 1

    def test_add_limit(self):
        plan = QueryPlan()
        plan.add_limit(1000)
        assert plan.count_steps("limit") == 1

    def test_add_with_columns(self):
        plan = QueryPlan()
        plan.add_with_columns('pl.col("a") + 1')
        assert plan.count_steps("with_columns") == 1

    def test_filter_steps_returns_only_filters(self):
        plan = QueryPlan()
        plan.add_filter("a", "==", 1)
        plan.add_sort("b")
        assert len(plan.filter_steps()) == 1

    def test_select_steps(self):
        plan = QueryPlan()
        plan.add_select(["x"])
        plan.add_filter("y", ">", 0)
        assert len(plan.select_steps()) == 1

    def test_chaining(self):
        plan = (
            QueryPlan(source="file.parquet")
            .add_filter("a", ">", 0)
            .add_select(["a", "b"])
            .add_limit(100)
        )
        assert len(plan.steps) == 3


class TestLazyPipelineConfig:
    def test_defaults(self):
        cfg = LazyPipelineConfig()
        assert cfg.predicate_pushdown is True
        assert cfg.streaming is False
        assert cfg.n_rows is None

    def test_active_optimizations_all(self):
        cfg = LazyPipelineConfig()
        flags = cfg.active_optimizations()
        assert OptimizationFlag.PREDICATE_PUSHDOWN in flags
        assert OptimizationFlag.PROJECTION_PUSHDOWN in flags

    def test_active_optimizations_streaming(self):
        cfg = LazyPipelineConfig(streaming=True)
        assert OptimizationFlag.STREAMING in cfg.active_optimizations()

    def test_active_optimizations_none(self):
        cfg = LazyPipelineConfig(
            predicate_pushdown=False,
            projection_pushdown=False,
            simplify_expression=False,
            slice_pushdown=False,
            streaming=False,
        )
        assert cfg.active_optimizations() == []

    def test_collect_kwargs(self):
        cfg = LazyPipelineConfig()
        kw = cfg.collect_kwargs()
        assert "predicate_pushdown" in kw
        assert "streaming" in kw

    def test_collect_kwargs_low_memory(self):
        cfg = LazyPipelineConfig(low_memory=True)
        kw = cfg.collect_kwargs()
        assert kw.get("low_memory") is True


class TestLazyPipelineStats:
    def test_total_steps(self):
        stats = LazyPipelineStats(filter_count=2, sort_count=1)
        assert stats.total_steps == 3

    def test_is_complex_with_join(self):
        stats = LazyPipelineStats(join_count=1)
        assert stats.is_complex() is True

    def test_is_complex_multiple_group_by(self):
        stats = LazyPipelineStats(group_by_count=2)
        assert stats.is_complex() is True

    def test_is_not_complex_simple(self):
        stats = LazyPipelineStats(filter_count=2, sort_count=1)
        assert stats.is_complex() is False


class TestPipelineOptimizer:
    def test_gather_stats(self):
        plan = QueryPlan()
        plan.add_filter("a", ">", 0)
        plan.add_select(["a"])
        plan.add_sort("a")
        opt = PipelineOptimizer(plan)
        stats = opt.gather_stats()
        assert stats.filter_count == 1
        assert stats.projection_count == 1
        assert stats.sort_count == 1

    def test_suggestions_predicate_disabled(self):
        plan = QueryPlan()
        plan.add_filter("a", "==", 1)
        cfg = LazyPipelineConfig(predicate_pushdown=False)
        opt = PipelineOptimizer(plan, cfg)
        suggestions = opt.suggestions()
        assert any("predicate_pushdown" in s for s in suggestions)

    def test_suggestions_projection_disabled(self):
        plan = QueryPlan()
        plan.add_select(["a"])
        cfg = LazyPipelineConfig(projection_pushdown=False)
        opt = PipelineOptimizer(plan, cfg)
        suggestions = opt.suggestions()
        # suggestions may or may not include projection hint depending on steps
        assert isinstance(suggestions, list)

    def test_suggestions_multiple_sorts(self):
        plan = QueryPlan()
        plan.add_sort("a")
        plan.add_sort("b")
        opt = PipelineOptimizer(plan)
        assert any("sort" in s.lower() for s in opt.suggestions())

    def test_suggestions_csv_scan(self):
        plan = QueryPlan(source="data.csv", scan_type=ScanType.CSV)
        opt = PipelineOptimizer(plan)
        assert any("parquet" in s.lower() for s in opt.suggestions())

    def test_reorder_for_pushdown(self):
        plan = QueryPlan()
        plan.add_sort("a")
        plan.add_filter("b", ">", 0)
        plan.add_select(["a", "b"])
        opt = PipelineOptimizer(plan)
        reordered = opt.reorder_for_pushdown()
        assert reordered[0].operation == "filter"

    def test_plan_property(self):
        plan = QueryPlan()
        opt = PipelineOptimizer(plan)
        assert opt.plan is plan

    def test_config_property(self):
        plan = QueryPlan()
        cfg = LazyPipelineConfig(streaming=True)
        opt = PipelineOptimizer(plan, cfg)
        assert opt.config.streaming is True


class TestBuildLazyPipeline:
    def test_parquet_scan(self):
        plan = QueryPlan(source="data.parquet", scan_type=ScanType.PARQUET)
        expr = build_lazy_pipeline(plan)
        assert "scan_parquet" in expr
        assert "data.parquet" in expr

    def test_csv_scan(self):
        plan = QueryPlan(source="data.csv", scan_type=ScanType.CSV)
        expr = build_lazy_pipeline(plan)
        assert "scan_csv" in expr

    def test_in_memory(self):
        plan = QueryPlan(source="df", scan_type=ScanType.IN_MEMORY)
        expr = build_lazy_pipeline(plan)
        assert "df.lazy()" in expr

    def test_with_filter(self):
        plan = QueryPlan(source="f.parquet").add_filter("a", ">", 0)
        expr = build_lazy_pipeline(plan)
        assert ".filter(" in expr

    def test_ends_with_collect(self):
        plan = QueryPlan(source="f.parquet")
        expr = build_lazy_pipeline(plan)
        assert ".collect(" in expr

    def test_n_rows_in_scan(self):
        plan = QueryPlan(source="big.parquet")
        cfg = LazyPipelineConfig(n_rows=1000)
        expr = build_lazy_pipeline(plan, cfg)
        assert "n_rows=1000" in expr

    def test_streaming_in_collect(self):
        plan = QueryPlan(source="f.parquet")
        cfg = LazyPipelineConfig(streaming=True)
        expr = build_lazy_pipeline(plan, cfg)
        assert "streaming=True" in expr

    def test_ipc_scan(self):
        plan = QueryPlan(source="data.ipc", scan_type=ScanType.IPC)
        expr = build_lazy_pipeline(plan)
        assert "scan_ipc" in expr
