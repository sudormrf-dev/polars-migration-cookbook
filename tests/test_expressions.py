"""Tests for expressions.py."""

from __future__ import annotations

import pytest

from patterns.expressions import (
    AggFunction,
    ExpressionBuilder,
    WindowFunction,
    WindowSpec,
    build_aggregation,
    build_filter,
    build_window_expr,
)


class TestBuildFilter:
    def test_empty_returns_lit_true(self):
        assert build_filter([]) == "pl.lit(True)"

    def test_single_eq(self):
        expr = build_filter([("age", "==", 30)])
        assert 'pl.col("age") == 30' in expr

    def test_single_ne(self):
        expr = build_filter([("status", "!=", "active")])
        assert "!= 'active'" in expr

    def test_single_gt(self):
        expr = build_filter([("score", ">", 0.5)])
        assert "> 0.5" in expr

    def test_single_gte(self):
        expr = build_filter([("score", ">=", 1)])
        assert ">= 1" in expr

    def test_single_lt(self):
        expr = build_filter([("n", "<", 100)])
        assert "< 100" in expr

    def test_single_lte(self):
        expr = build_filter([("n", "<=", 50)])
        assert "<= 50" in expr

    def test_in_operator(self):
        expr = build_filter([("cat", "in", ["a", "b"])])
        assert ".is_in" in expr

    def test_not_in_operator(self):
        expr = build_filter([("cat", "not_in", ["x"])])
        assert "~" in expr
        assert ".is_in" in expr

    def test_unknown_operator_passthrough(self):
        expr = build_filter([("col", "UNKNOWN", None)])
        assert 'pl.col("col")' in expr

    def test_and_combine(self):
        conditions = [("a", "==", 1), ("b", ">", 2)]
        expr = build_filter(conditions, combine="and")
        assert " & " in expr

    def test_or_combine(self):
        conditions = [("a", "==", 1), ("b", "==", 2)]
        expr = build_filter(conditions, combine="or")
        assert " | " in expr

    def test_invalid_combine_raises(self):
        with pytest.raises(ValueError):
            build_filter([("a", "==", 1)], combine="xor")

    def test_single_condition_no_parens(self):
        expr = build_filter([("x", "==", 5)])
        assert not expr.startswith("(")

    def test_multi_condition_has_parens(self):
        expr = build_filter([("x", "==", 1), ("y", "==", 2)])
        assert expr.startswith("(")


class TestBuildAggregation:
    def test_basic(self):
        expr = build_aggregation(["dept"], {"salary": AggFunction.MEAN})
        assert "group_by" in expr
        assert "mean()" in expr

    def test_alias(self):
        expr = build_aggregation(
            ["dept"],
            {"salary": AggFunction.SUM},
            aliases={"salary": "total_salary"},
        )
        assert 'alias("total_salary")' in expr

    def test_multiple_columns(self):
        expr = build_aggregation(
            ["region"],
            {"sales": AggFunction.SUM, "count": AggFunction.COUNT},
        )
        assert "sum()" in expr
        assert "count()" in expr

    def test_group_by_multiple_keys(self):
        expr = build_aggregation(["a", "b"], {"val": AggFunction.MAX})
        assert '"a"' in expr
        assert '"b"' in expr

    def test_all_agg_functions(self):
        for fn in AggFunction:
            expr = build_aggregation(["g"], {"v": fn})
            assert fn.value in expr


class TestWindowSpec:
    def test_is_valid_with_partition(self):
        spec = WindowSpec(partition_by=["user_id"])
        assert spec.is_valid() is True

    def test_is_valid_with_order(self):
        spec = WindowSpec(order_by="ts")
        assert spec.is_valid() is True

    def test_is_not_valid_empty(self):
        spec = WindowSpec()
        assert spec.is_valid() is False

    def test_defaults(self):
        spec = WindowSpec()
        assert spec.window_size == 3
        assert spec.offset == 1
        assert spec.descending is False


class TestBuildWindowExpr:
    def setup_method(self):
        self.spec = WindowSpec(partition_by=["user_id"], order_by="ts")

    def test_rolling_mean(self):
        expr = build_window_expr("value", WindowFunction.ROLLING_MEAN, self.spec)
        assert "rolling_mean" in expr
        assert "window_size=" in expr
        assert "over(" in expr

    def test_rolling_sum(self):
        expr = build_window_expr("value", "rolling_sum", self.spec)
        assert "rolling_sum" in expr

    def test_lag(self):
        expr = build_window_expr("value", WindowFunction.LAG, self.spec)
        assert "shift(" in expr
        assert "over(" in expr

    def test_lead(self):
        expr = build_window_expr("value", "lead", self.spec)
        assert "shift(" in expr

    def test_rank(self):
        expr = build_window_expr("value", WindowFunction.RANK, self.spec)
        assert "rank(" in expr
        assert 'method="rank"' in expr

    def test_dense_rank(self):
        expr = build_window_expr("value", "dense_rank", self.spec)
        assert 'method="dense_rank"' in expr

    def test_row_number(self):
        expr = build_window_expr("value", WindowFunction.ROW_NUMBER, self.spec)
        assert "row_number" in expr

    def test_cumsum(self):
        expr = build_window_expr("value", WindowFunction.CUMSUM, self.spec)
        assert "cumsum()" in expr

    def test_cumprod(self):
        expr = build_window_expr("value", "cumprod", self.spec)
        assert "cumprod()" in expr

    def test_rank_descending(self):
        spec = WindowSpec(partition_by=["u"], order_by="score", descending=True)
        expr = build_window_expr("score", "rank", spec)
        assert "sort(descending=True)" in expr

    def test_empty_partition(self):
        spec = WindowSpec(order_by="ts")
        expr = build_window_expr("v", "cumsum", spec)
        assert "over([])" in expr

    def test_string_function_passthrough(self):
        expr = build_window_expr("v", "custom_fn", self.spec)
        assert "custom_fn()" in expr


class TestExpressionBuilder:
    def test_bare_col(self):
        e = ExpressionBuilder("price")
        assert e.build() == 'pl.col("price")'

    def test_cast(self):
        e = ExpressionBuilder("price").cast("Float64")
        assert "cast(pl.Float64)" in e.build()

    def test_clip_lower(self):
        e = ExpressionBuilder("score").clip(lower=0.0)
        assert "lower_bound=0.0" in e.build()

    def test_clip_upper(self):
        e = ExpressionBuilder("score").clip(upper=100.0)
        assert "upper_bound=100.0" in e.build()

    def test_clip_both(self):
        e = ExpressionBuilder("score").clip(lower=0.0, upper=1.0)
        assert "lower_bound=0.0" in e.build()
        assert "upper_bound=1.0" in e.build()

    def test_fill_null(self):
        e = ExpressionBuilder("col").fill_null(0)
        assert "fill_null(0)" in e.build()

    def test_alias(self):
        e = ExpressionBuilder("col").alias("new_name")
        assert 'alias("new_name")' in e.build()

    def test_str_replace(self):
        e = ExpressionBuilder("text").str_replace("foo", "bar")
        assert "str.replace_all" in e.build()

    def test_chain(self):
        e = ExpressionBuilder("price").clip(lower=0.0).fill_null(0.0).alias("clean_price")
        result = e.build()
        assert "clip(" in result
        assert "fill_null(" in result
        assert "alias(" in result

    def test_column_property(self):
        e = ExpressionBuilder("myCol")
        assert e.column == "myCol"

    def test_fluent_returns_self(self):
        e = ExpressionBuilder("x")
        assert e.cast("Int32") is e

    def test_multiple_ops_order(self):
        e = ExpressionBuilder("x").cast("Int32").fill_null(0).alias("y")
        result = e.build()
        ops = result.split(".")
        cast_idx = next(i for i, o in enumerate(ops) if "cast" in o)
        fill_idx = next(i for i, o in enumerate(ops) if "fill_null" in o)
        alias_idx = next(i for i, o in enumerate(ops) if "alias" in o)
        assert cast_idx < fill_idx < alias_idx
