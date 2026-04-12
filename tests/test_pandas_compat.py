"""Tests for pandas_compat.py."""

from __future__ import annotations

import pytest

from patterns.pandas_compat import (
    ConversionConfig,
    DataFrameConverter,
    MigrationWarning,
    SchemaMapper,
    infer_polars_dtype,
)


class TestInferPolarsType:
    def test_int64(self):
        assert infer_polars_dtype("int64") == "Int64"

    def test_float32(self):
        assert infer_polars_dtype("float32") == "Float32"

    def test_object(self):
        assert infer_polars_dtype("object") == "Utf8"

    def test_bool(self):
        assert infer_polars_dtype("bool") == "Boolean"

    def test_category(self):
        assert infer_polars_dtype("category") == "Categorical"

    def test_datetime_ns(self):
        assert infer_polars_dtype("datetime64[ns]") == "Datetime"

    def test_unknown(self):
        assert infer_polars_dtype("custom_type") == "Unknown"

    def test_case_insensitive(self):
        assert infer_polars_dtype("INT64") == "Int64"

    def test_string(self):
        assert infer_polars_dtype("string") == "Utf8"

    def test_uint32(self):
        assert infer_polars_dtype("uint32") == "UInt32"


class TestSchemaMapper:
    def test_to_polars_schema_basic(self):
        m = SchemaMapper(pandas_dtypes={"age": "int64", "name": "object"})
        schema = m.to_polars_schema()
        assert schema["age"] == "Int64"
        assert schema["name"] == "Utf8"

    def test_unmapped_columns(self):
        m = SchemaMapper(pandas_dtypes={"a": "int64", "b": "custom"})
        assert m.unmapped_columns() == ["b"]

    def test_no_unmapped(self):
        m = SchemaMapper(pandas_dtypes={"x": "float64"})
        assert m.unmapped_columns() == []

    def test_has_object_columns_true(self):
        m = SchemaMapper(pandas_dtypes={"col": "object"})
        assert m.has_object_columns() is True

    def test_has_object_columns_false(self):
        m = SchemaMapper(pandas_dtypes={"col": "int64"})
        assert m.has_object_columns() is False

    def test_empty_schema(self):
        m = SchemaMapper()
        assert m.to_polars_schema() == {}
        assert m.unmapped_columns() == []


class TestConversionConfig:
    def test_defaults(self):
        cfg = ConversionConfig()
        assert cfg.coerce_timestamps is True
        assert cfg.rechunk is True
        assert cfg.nan_to_null is True
        assert cfg.infer_schema_length == 100

    def test_custom(self):
        cfg = ConversionConfig(rechunk=False, string_cache=True)
        assert cfg.rechunk is False
        assert cfg.string_cache is True


class TestDataFrameConverter:
    def setup_method(self):
        self.converter = DataFrameConverter()

    def test_config_default(self):
        assert self.converter.config.coerce_timestamps is True

    def test_audit_named_index(self):
        issues = self.converter.audit_dataframe({"has_named_index": True})
        types = [w for w, _ in issues]
        assert MigrationWarning.IMPLICIT_INDEX in types

    def test_audit_apply(self):
        issues = self.converter.audit_dataframe({"uses_apply": True})
        types = [w for w, _ in issues]
        assert MigrationWarning.APPLY_USAGE in types

    def test_audit_inplace(self):
        issues = self.converter.audit_dataframe({"uses_inplace": True})
        types = [w for w, _ in issues]
        assert MigrationWarning.INPLACE_MUTATION in types

    def test_audit_object_dtype(self):
        issues = self.converter.audit_dataframe({"dtypes": {"name": "object"}})
        types = [w for w, _ in issues]
        assert MigrationWarning.OBJECT_DTYPE in types

    def test_audit_clean(self):
        issues = self.converter.audit_dataframe({"has_named_index": False, "uses_apply": False})
        assert issues == []

    def test_migration_checklist_non_empty(self):
        checklist = self.converter.migration_checklist()
        assert len(checklist) > 0
        assert any("group_by" in item for item in checklist)

    def test_custom_config(self):
        cfg = ConversionConfig(rechunk=False)
        conv = DataFrameConverter(cfg)
        assert conv.config.rechunk is False

    def test_audit_multiple_issues(self):
        info = {
            "has_named_index": True,
            "uses_apply": True,
            "dtypes": {"col": "object"},
        }
        issues = self.converter.audit_dataframe(info)
        assert len(issues) >= 3

    def test_audit_empty_dict(self):
        issues = self.converter.audit_dataframe({})
        assert issues == []

    @pytest.mark.parametrize("dtype", ["Object", "OBJECT", "object"])
    def test_audit_object_dtype_case_variants(self, dtype: str):
        issues = self.converter.audit_dataframe({"dtypes": {"col": dtype}})
        assert len(issues) >= 1
