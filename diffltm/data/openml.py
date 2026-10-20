"""Adult acquisition from its pinned OpenML ID, separate from shared splitting."""

import pandas as pd
from sklearn.datasets import fetch_openml

from .core import Schema, make_table
from .openml_registry import OPENML_REGISTRY


def build(name, cache):
    spec = OPENML_REGISTRY[name]
    bunch = fetch_openml(
        data_id=spec.openml_id, as_frame=True, data_home=str(cache) + "/openml", parser="auto"
    )
    df = bunch.data.copy()
    if spec.target not in df.columns:
        if getattr(bunch.target, "ndim", 1) != 1:
            raise ValueError("Registry must identify one target for multi-target OpenML data")
        df[spec.target] = bunch.target.to_numpy()
    df = df.rename(columns=spec.rename_cols).drop(columns=spec.drop_cols)
    for c in df:
        df[c] = df[c].astype(object) if isinstance(df[c].dtype, pd.CategoricalDtype) else df[c]
        df[c] = df[c].replace(spec.missing_tokens, float("nan"))
    if df[spec.target].isna().any():
        df = df.loc[df[spec.target].notna()].copy()
    if spec.all_categorical:
        num, cat = [], list(df)
    elif spec.numeric_cols is not None or spec.categorical_cols is not None:
        num = (
            spec.numeric_cols
            if spec.numeric_cols is not None
            else [c for c in df if c not in spec.categorical_cols]
        )
        cat = (
            spec.categorical_cols
            if spec.categorical_cols is not None
            else [c for c in df if c not in num]
        )
    else:
        num = [c for c in df if pd.api.types.is_numeric_dtype(df[c]) and c != spec.target]
        cat = [c for c in df if c not in num and c != spec.target]
        (num if spec.task == "regression" else cat).append(spec.target)
    for c in num:
        df[c] = pd.to_numeric(df[c])
    return make_table(
        df,
        Schema(num, cat, spec.target, spec.task),
        name,
        {"downloader": "sklearn.fetch_openml", "data_id": spec.openml_id},
    )
