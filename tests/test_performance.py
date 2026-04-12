"""Tests for performance.py."""

from __future__ import annotations

import time

from patterns.performance import (
    BenchmarkResult,
    BottleneckType,
    MemoryProfile,
    ParallelConfig,
    PerformanceAdvisor,
    Timer,
    profile_memory,
)


class TestBenchmarkResult:
    def test_rows_per_second(self):
        r = BenchmarkResult(label="test", elapsed_seconds=2.0, rows_processed=1_000_000)
        assert r.rows_per_second == 500_000.0

    def test_rows_per_second_zero_elapsed(self):
        r = BenchmarkResult(label="test", elapsed_seconds=0.0, rows_processed=100)
        assert r.rows_per_second == 0.0

    def test_ms(self):
        r = BenchmarkResult(label="t", elapsed_seconds=1.5)
        assert r.ms == 1500.0

    def test_is_faster_than_true(self):
        fast = BenchmarkResult(label="fast", elapsed_seconds=0.1)
        slow = BenchmarkResult(label="slow", elapsed_seconds=1.0)
        assert fast.is_faster_than(slow) is True

    def test_is_faster_than_false(self):
        fast = BenchmarkResult(label="fast", elapsed_seconds=0.1)
        slow = BenchmarkResult(label="slow", elapsed_seconds=1.0)
        assert slow.is_faster_than(fast) is False

    def test_speedup_vs(self):
        fast = BenchmarkResult(label="fast", elapsed_seconds=0.5)
        slow = BenchmarkResult(label="slow", elapsed_seconds=2.0)
        assert fast.speedup_vs(slow) == 4.0

    def test_speedup_vs_zero(self):
        r1 = BenchmarkResult(label="a", elapsed_seconds=0.0)
        r2 = BenchmarkResult(label="b", elapsed_seconds=1.0)
        assert r1.speedup_vs(r2) == 1.0


class TestProfileMemory:
    def test_basic_schema(self):
        schema = {"age": "Int32", "name": "Utf8"}
        profile = profile_memory(schema, n_rows=1000)
        assert profile.n_rows == 1000
        assert profile.total_bytes > 0

    def test_int32_size(self):
        profile = profile_memory({"x": "Int32"}, n_rows=100)
        assert profile.per_column_bytes["x"] == 400  # 4 bytes * 100

    def test_utf8_size(self):
        profile = profile_memory({"s": "Utf8"}, n_rows=10)
        assert profile.per_column_bytes["s"] == 320  # 32 bytes * 10

    def test_total_is_sum(self):
        schema = {"a": "Int64", "b": "Float64"}
        profile = profile_memory(schema, n_rows=100)
        assert profile.total_bytes == sum(profile.per_column_bytes.values())

    def test_unknown_dtype_default(self):
        profile = profile_memory({"x": "CustomType"}, n_rows=1)
        assert profile.per_column_bytes["x"] == 8


class TestMemoryProfile:
    def setup_method(self):
        self.profile = MemoryProfile(
            schema={"a": "Int64", "b": "Utf8"},
            n_rows=1_000_000,
            per_column_bytes={"a": 8_000_000, "b": 32_000_000},
            total_bytes=40_000_000,
        )

    def test_total_mb(self):
        mb = self.profile.total_mb
        assert mb > 0

    def test_total_gb(self):
        gb = self.profile.total_gb
        assert 0 < gb < 1

    def test_largest_columns(self):
        largest = self.profile.largest_columns(1)
        assert largest[0][0] == "b"

    def test_needs_streaming_false(self):
        assert self.profile.needs_streaming(available_ram_gb=8.0) is False

    def test_needs_streaming_true(self):
        big = MemoryProfile(total_bytes=10 * 1024 * 1024 * 1024)  # 10 GB
        assert big.needs_streaming(available_ram_gb=8.0) is True

    def test_largest_columns_respects_top_n(self):
        assert len(self.profile.largest_columns(1)) == 1


class TestParallelConfig:
    def test_defaults(self):
        cfg = ParallelConfig()
        assert cfg.n_threads == 0
        assert cfg.use_streaming is False

    def test_env_vars_zero_threads(self):
        cfg = ParallelConfig(n_threads=0)
        assert "POLARS_MAX_THREADS" not in cfg.env_vars()

    def test_env_vars_nonzero_threads(self):
        cfg = ParallelConfig(n_threads=8)
        assert cfg.env_vars()["POLARS_MAX_THREADS"] == "8"

    def test_streaming_recommended_large(self):
        cfg = ParallelConfig()
        big = MemoryProfile(total_bytes=5 * 1024 * 1024 * 1024)  # 5 GB
        assert cfg.is_streaming_recommended(big) is True

    def test_streaming_recommended_forced(self):
        cfg = ParallelConfig(use_streaming=True)
        small = MemoryProfile(total_bytes=1024)
        assert cfg.is_streaming_recommended(small) is True

    def test_streaming_not_recommended(self):
        cfg = ParallelConfig()
        small = MemoryProfile(total_bytes=1024)
        assert cfg.is_streaming_recommended(small) is False


class TestPerformanceAdvisor:
    def test_no_issues_clean_schema(self):
        advisor = PerformanceAdvisor(schema={"age": "Int32", "score": "Float64"})
        issues = advisor.detect_bottlenecks()
        assert issues == []

    def test_apply_usage_detected(self):
        advisor = PerformanceAdvisor(schema={}, usage={"uses_apply": True})
        types = [t for t, _ in advisor.detect_bottlenecks()]
        assert BottleneckType.APPLY_USAGE in types

    def test_utf8_column_flagged(self):
        advisor = PerformanceAdvisor(schema={"name": "Utf8"})
        types = [t for t, _ in advisor.detect_bottlenecks()]
        assert BottleneckType.HIGH_CARDINALITY_STRING in types

    def test_string_column_flagged(self):
        advisor = PerformanceAdvisor(schema={"tag": "String"})
        types = [t for t, _ in advisor.detect_bottlenecks()]
        assert BottleneckType.HIGH_CARDINALITY_STRING in types

    def test_eager_wide_schema(self):
        schema = {f"col_{i}": "Int64" for i in range(10)}
        advisor = PerformanceAdvisor(schema=schema, usage={"uses_eager": True})
        types = [t for t, _ in advisor.detect_bottlenecks()]
        assert BottleneckType.MISSING_LAZY in types

    def test_sort_before_filter(self):
        advisor = PerformanceAdvisor(schema={}, usage={"sort_before_filter": True})
        types = [t for t, _ in advisor.detect_bottlenecks()]
        assert BottleneckType.SORT_BEFORE_FILTER in types

    def test_concat_fragmentation(self):
        advisor = PerformanceAdvisor(schema={}, usage={"concat_count": 5})
        types = [t for t, _ in advisor.detect_bottlenecks()]
        assert BottleneckType.FRAGMENTED_MEMORY in types

    def test_concat_below_threshold_no_flag(self):
        advisor = PerformanceAdvisor(schema={}, usage={"concat_count": 2})
        types = [t for t, _ in advisor.detect_bottlenecks()]
        assert BottleneckType.FRAGMENTED_MEMORY not in types

    def test_repeated_scan(self):
        advisor = PerformanceAdvisor(schema={}, usage={"scan_count": 3})
        types = [t for t, _ in advisor.detect_bottlenecks()]
        assert BottleneckType.REPEATED_SCAN in types

    def test_object_dtype_flagged(self):
        advisor = PerformanceAdvisor(schema={"col": "object"})
        types = [t for t, _ in advisor.detect_bottlenecks()]
        assert BottleneckType.OBJECT_DTYPE in types

    def test_advise_returns_strings(self):
        advisor = PerformanceAdvisor(schema={"n": "Utf8"})
        advice = advisor.advise()
        assert all(isinstance(a, str) for a in advice)

    def test_dtype_summary(self):
        schema = {"a": "Int64", "b": "Int64", "c": "Utf8"}
        advisor = PerformanceAdvisor(schema=schema)
        summary = advisor.dtype_summary()
        assert summary["Int64"] == 2
        assert summary["Utf8"] == 1


class TestTimer:
    def test_records_result(self):
        with Timer("test", rows=100) as t:
            time.sleep(0.001)
        assert t.result is not None
        assert t.result.elapsed_seconds > 0

    def test_label_preserved(self):
        with Timer("my_label") as t:
            pass
        assert t.result is not None
        assert t.result.label == "my_label"

    def test_rows_preserved(self):
        with Timer("x", rows=500) as t:
            pass
        assert t.result is not None
        assert t.result.rows_processed == 500

    def test_elapsed_positive(self):
        with Timer("t") as t:
            pass
        assert t.result is not None
        assert t.result.elapsed_seconds >= 0
