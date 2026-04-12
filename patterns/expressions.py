"""Polars expression patterns: filters, aggregations, window functions.

Polars expressions are lazy computation graphs evaluated in parallel.
Key advantage over pandas: expressions compose without materializing
intermediate DataFrames.

Patterns:
  - build_filter(): combine multiple column conditions with AND/OR
  - build_aggregation(): group_by + named agg dict
  - build_window_expr(): rolling/rank/lag window functions
  - ExpressionBuilder: fluent interface for common transforms

Usage::

    spec = WindowSpec(partition_by=["user_id"], order_by="timestamp")
    expr_str = build_window_expr("value", "rank", spec)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AggFunction(str, Enum):
    """Supported aggregation functions."""

    SUM = "sum"
    MEAN = "mean"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    FIRST = "first"
    LAST = "last"
    STD = "std"
    VAR = "var"
    MEDIAN = "median"
    N_UNIQUE = "n_unique"


class WindowFunction(str, Enum):
    """Supported window functions."""

    RANK = "rank"
    DENSE_RANK = "dense_rank"
    ROW_NUMBER = "row_number"
    LAG = "lag"
    LEAD = "lead"
    ROLLING_MEAN = "rolling_mean"
    ROLLING_SUM = "rolling_sum"
    CUMSUM = "cumsum"
    CUMPROD = "cumprod"


@dataclass
class WindowSpec:
    """Specification for a window function.

    Attributes:
        partition_by: Columns to partition by (equivalent to GROUP BY).
        order_by: Column to order by within the partition.
        descending: Order direction.
        window_size: For rolling functions, the window size.
        offset: For lag/lead, the number of rows to offset.
    """

    partition_by: list[str] = field(default_factory=list)
    order_by: str = ""
    descending: bool = False
    window_size: int = 3
    offset: int = 1

    def is_valid(self) -> bool:
        """Return True if the spec has at least a partition or order."""
        return bool(self.partition_by) or bool(self.order_by)


def build_filter(
    conditions: list[tuple[str, str, Any]],
    combine: str = "and",
) -> str:
    """Build a filter expression string from condition tuples.

    Args:
        conditions: List of (column, operator, value) tuples.
            Operators: "==", "!=", ">", ">=", "<", "<=", "in", "not_in".
        combine: "and" or "or" to combine multiple conditions.

    Returns:
        Polars expression string (for documentation/codegen).

    Raises:
        ValueError: If combine is not "and" or "or".
    """
    if combine not in {"and", "or"}:
        err = f"combine must be 'and' or 'or', got {combine!r}"
        raise ValueError(err)

    parts: list[str] = []
    for col, op, val in conditions:
        if op == "==":
            parts.append(f'pl.col("{col}") == {val!r}')
        elif op == "!=":
            parts.append(f'pl.col("{col}") != {val!r}')
        elif op == ">":
            parts.append(f'pl.col("{col}") > {val!r}')
        elif op == ">=":
            parts.append(f'pl.col("{col}") >= {val!r}')
        elif op == "<":
            parts.append(f'pl.col("{col}") < {val!r}')
        elif op == "<=":
            parts.append(f'pl.col("{col}") <= {val!r}')
        elif op == "in":
            parts.append(f'pl.col("{col}").is_in({val!r})')
        elif op == "not_in":
            parts.append(f'~pl.col("{col}").is_in({val!r})')
        else:
            parts.append(f'pl.col("{col}")')

    if not parts:
        return "pl.lit(True)"

    joiner = " & " if combine == "and" else " | "
    if len(parts) == 1:
        return parts[0]
    return "(" + joiner.join(parts) + ")"


def build_aggregation(
    group_by: list[str],
    aggs: dict[str, AggFunction],
    aliases: dict[str, str] | None = None,
) -> str:
    """Build a group_by aggregation expression string.

    Args:
        group_by: Columns to group by.
        aggs: Dict of column → aggregation function.
        aliases: Optional dict of column → output alias.

    Returns:
        Polars expression string showing .group_by().agg() pattern.
    """
    alias_map = aliases or {}
    agg_parts: list[str] = []
    for col, fn in aggs.items():
        expr = f'pl.col("{col}").{fn.value}()'
        if col in alias_map:
            expr += f'.alias("{alias_map[col]}")'
        agg_parts.append(expr)

    group_str = ", ".join(f'"{c}"' for c in group_by)
    agg_str = ", ".join(agg_parts)
    return f"df.group_by({group_str}).agg({agg_str})"


def build_window_expr(
    column: str,
    function: str | WindowFunction,
    spec: WindowSpec,
) -> str:
    """Build a window function expression string.

    Args:
        column: Column to apply the window function to.
        function: Window function name or :class:`WindowFunction`.
        spec: Window specification.

    Returns:
        Polars expression string for the window function.
    """
    fn = function.value if isinstance(function, WindowFunction) else function
    partition_str = (
        "[" + ", ".join(f'"{c}"' for c in spec.partition_by) + "]" if spec.partition_by else "[]"
    )

    if fn in {"rolling_mean", "rolling_sum"}:
        return f'pl.col("{column}").{fn}(window_size={spec.window_size}).over({partition_str})'
    if fn in {"lag", "lead"}:
        return f'pl.col("{column}").shift({spec.offset}).over({partition_str})'
    if fn in {"rank", "dense_rank", "row_number"}:
        order = f'pl.col("{spec.order_by}")'
        if spec.descending:
            order += ".sort(descending=True)"
        return f'{order}.rank(method="{fn}").over({partition_str})'
    if fn in {"cumsum", "cumprod"}:
        return f'pl.col("{column}").{fn}().over({partition_str})'

    return f'pl.col("{column}").{fn}().over({partition_str})'


class ExpressionBuilder:
    """Fluent builder for common Polars expression patterns.

    Args:
        column: Target column name.
    """

    def __init__(self, column: str) -> None:
        self._col = column
        self._ops: list[str] = []

    def cast(self, dtype: str) -> ExpressionBuilder:
        """Add a cast operation."""
        self._ops.append(f"cast(pl.{dtype})")
        return self

    def clip(self, lower: float | None = None, upper: float | None = None) -> ExpressionBuilder:
        """Add a clip operation."""
        args = []
        if lower is not None:
            args.append(f"lower_bound={lower}")
        if upper is not None:
            args.append(f"upper_bound={upper}")
        self._ops.append(f"clip({', '.join(args)})")
        return self

    def fill_null(self, value: Any) -> ExpressionBuilder:
        """Add a fill_null operation."""
        self._ops.append(f"fill_null({value!r})")
        return self

    def alias(self, name: str) -> ExpressionBuilder:
        """Add an alias."""
        self._ops.append(f'alias("{name}")')
        return self

    def str_replace(self, pattern: str, replacement: str) -> ExpressionBuilder:
        """Add str.replace operation."""
        self._ops.append(f'str.replace_all("{pattern}", "{replacement}")')
        return self

    def build(self) -> str:
        """Return the expression string."""
        base = f'pl.col("{self._col}")'
        if not self._ops:
            return base
        return base + "." + ".".join(self._ops)

    @property
    def column(self) -> str:
        """The target column name."""
        return self._col
