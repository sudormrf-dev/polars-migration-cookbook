"""pandas → Polars compatibility and migration helpers.

Key differences from pandas:
- No implicit index (use explicit row_number() if needed)
- Strings are Utf8, not object
- Nulls are null/None, not NaN (for non-float types)
- .apply() is replaced by map_elements() — but avoid it for performance
- groupby() → group_by(), rename() uses dict not mapper
- inplace= does not exist

Usage::

    mapper = SchemaMapper(pandas_dtypes={"age": "int64", "name": "object"})
    polars_schema = mapper.to_polars_schema()

    config = ConversionConfig(coerce_timestamps=True, rechunk=True)
    converter = DataFrameConverter(config)
    warnings = converter.audit_dataframe(df_dict)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MigrationWarning(str, Enum):
    """Warning category emitted during pandas→Polars migration analysis."""

    IMPLICIT_INDEX = "implicit_index"
    APPLY_USAGE = "apply_usage"
    INPLACE_MUTATION = "inplace_mutation"
    OBJECT_DTYPE = "object_dtype"
    NAN_VS_NULL = "nan_vs_null"
    CHAINED_INDEXING = "chained_indexing"
    DEPRECATED_API = "deprecated_api"


# pandas dtype → Polars dtype string mapping
_PANDAS_TO_POLARS: dict[str, str] = {
    "int8": "Int8",
    "int16": "Int16",
    "int32": "Int32",
    "int64": "Int64",
    "uint8": "UInt8",
    "uint16": "UInt16",
    "uint32": "UInt32",
    "uint64": "UInt64",
    "float32": "Float32",
    "float64": "Float64",
    "bool": "Boolean",
    "object": "Utf8",
    "string": "Utf8",
    "category": "Categorical",
    "datetime64[ns]": "Datetime",
    "datetime64[us]": "Datetime[us]",
    "timedelta64[ns]": "Duration",
    "date": "Date",
}


def infer_polars_dtype(pandas_dtype: str) -> str:
    """Map a pandas dtype string to a Polars dtype string.

    Args:
        pandas_dtype: pandas dtype as a string (e.g. "int64", "object").

    Returns:
        Polars dtype string, or "Unknown" if not mappable.
    """
    return _PANDAS_TO_POLARS.get(pandas_dtype.lower(), "Unknown")


@dataclass
class SchemaMapper:
    """Maps pandas column dtypes to Polars schema.

    Args:
        pandas_dtypes: Dict of column_name → pandas dtype string.
    """

    pandas_dtypes: dict[str, str] = field(default_factory=dict)

    def to_polars_schema(self) -> dict[str, str]:
        """Return {column: polars_dtype} for all columns."""
        return {col: infer_polars_dtype(dtype) for col, dtype in self.pandas_dtypes.items()}

    def unmapped_columns(self) -> list[str]:
        """Return column names whose dtype could not be mapped."""
        return [
            col
            for col, dtype in self.pandas_dtypes.items()
            if infer_polars_dtype(dtype) == "Unknown"
        ]

    def has_object_columns(self) -> bool:
        """True if any column has pandas object dtype."""
        return any(d.lower() == "object" for d in self.pandas_dtypes.values())


@dataclass
class ConversionConfig:
    """Configuration for pandas→Polars DataFrame conversion.

    Attributes:
        coerce_timestamps: Convert datetime columns to Polars Datetime.
        rechunk: Rechunk after conversion for memory efficiency.
        infer_schema_length: Rows to scan for schema inference (0 = all).
        nan_to_null: Replace float NaN with null during conversion.
        string_cache: Enable global string cache for Categorical columns.
    """

    coerce_timestamps: bool = True
    rechunk: bool = True
    infer_schema_length: int = 100
    nan_to_null: bool = True
    string_cache: bool = False


class DataFrameConverter:
    """Audits pandas DataFrame structures for migration readiness.

    Does not import pandas or Polars — works on dict representations
    for testability without those dependencies.

    Args:
        config: Conversion configuration.
    """

    def __init__(self, config: ConversionConfig | None = None) -> None:
        self._cfg = config or ConversionConfig()

    @property
    def config(self) -> ConversionConfig:
        """The conversion configuration."""
        return self._cfg

    def audit_dataframe(self, df_info: dict[str, Any]) -> list[tuple[MigrationWarning, str]]:
        """Audit a DataFrame description for migration issues.

        Args:
            df_info: Dict with keys like 'columns', 'has_index',
                     'uses_apply', 'dtypes'.

        Returns:
            List of (warning_type, message) tuples.
        """
        issues: list[tuple[MigrationWarning, str]] = []

        if df_info.get("has_named_index"):
            issues.append(
                (
                    MigrationWarning.IMPLICIT_INDEX,
                    "Named index detected — use explicit 'row_nr' column in Polars",
                )
            )

        if df_info.get("uses_apply"):
            issues.append(
                (
                    MigrationWarning.APPLY_USAGE,
                    ".apply()/.map() detected — replace with Polars expressions or map_elements()",
                )
            )

        if df_info.get("uses_inplace"):
            issues.append(
                (
                    MigrationWarning.INPLACE_MUTATION,
                    "inplace=True detected — Polars is immutable, reassign the result",
                )
            )

        dtypes = df_info.get("dtypes", {})
        for col, dtype in dtypes.items():
            if str(dtype).lower() == "object":
                issues.append(
                    (
                        MigrationWarning.OBJECT_DTYPE,
                        f"Column '{col}' has object dtype — use Utf8/Categorical in Polars",
                    )
                )

        return issues

    def migration_checklist(self) -> list[str]:
        """Return a migration checklist for pandas→Polars."""
        return [
            "Replace df.reset_index() with pl.DataFrame (no implicit index)",
            "Replace df.apply(fn) with pl.Expr.map_elements(fn) or native expressions",
            "Replace df[col].str.contains(pat) with pl.col(col).str.contains(pat)",
            "Replace df.groupby(col).agg() with df.group_by(col).agg()",
            "Replace df.merge() with df.join()",
            "Replace df.fillna(val) with df.fill_null(val)",
            "Replace df.dropna() with df.drop_nulls()",
            "Use lazy evaluation: df.lazy().filter().collect()",
        ]
