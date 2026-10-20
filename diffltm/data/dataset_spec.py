from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True, kw_only=True)
class DatasetSpec:
    """
    Thin base interface for dataset metadata + minimal cleanup/schema hints.
    """

    name: str
    target: str
    task: Literal["binclass", "multiclass", "regression"] = "binclass"
    n_classes: int  # For classification tasks; set to 0 for regression

    # Optional schema overrides (None => infer)
    numeric_cols: list[str] | None = None
    categorical_cols: list[str] | None = None
    # If True, treat ALL (post-drop) columns as categorical regardless of dtype.
    # Use for integer-coded categorical datasets where dtype
    # inference would wrongly type them as numerical. Overrides numeric/categorical_cols.
    all_categorical: bool = False

    # Minimal cleaning hooks
    rename_cols: dict[str, str] = field(default_factory=dict)
    drop_cols: list[str] = field(default_factory=list)
    missing_tokens: list[object] = field(default_factory=lambda: ["?"])

    # Informational only (logging / docs)
    description: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class OpenMLDatasetSpec(DatasetSpec):
    openml_id: int
