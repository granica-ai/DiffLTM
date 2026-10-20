"""Train-fitted numeric maps and ordered categorical codes with unknown ID 0."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from ..artifacts import identity
from .core import category_values


def fit_numeric_fill(train, schema):
    num = train[schema.numerical].to_numpy(float)
    if np.isinf(num).any():
        raise ValueError("Infinite numeric training value")
    fill = np.nanmean(num, axis=0)
    if not np.isfinite(fill).all():
        raise ValueError("All-missing numerical column cannot be fitted")
    return fill


def imputed_frame(frame, schema, fill):
    """Membership and scoring view in raw units, using training-fitted means."""
    return frame.fillna(dict(zip(schema.numerical, fill)))


def imputation_identity(schema, fill):
    return identity(["train-mean-v1", schema.numerical, np.asarray(fill).tolist()])


class Preprocessor:
    def __init__(self, schema, numerical="quantile", seed=0):
        self.schema, self.numerical, self.seed = schema, numerical, seed

    def fit(self, train):
        num = train[self.schema.numerical].to_numpy(float)
        self.fill = fit_numeric_fill(train, self.schema)
        num = np.where(np.isnan(num), self.fill, num)
        self.mean = num.mean(0)
        self.scale = num.std(0, ddof=0)
        self.scale[self.scale == 0] = 1
        self.raw_min, self.raw_max = num.min(0), num.max(0)
        if self.numerical not in {"quantile", "standardize"}:
            raise ValueError(self.numerical)
        self.transformer = (
            (
                QuantileTransformer(
                    n_quantiles=min(1000, len(train)),
                    output_distribution="normal",
                    subsample=None,
                    random_state=self.seed,
                )
                if self.numerical == "quantile"
                else StandardScaler()
            )
            if num.shape[1]
            else None
        )
        modeled = self.transformer.fit_transform(num) if self.transformer is not None else num
        self.model_min, self.model_max = modeled.min(0), modeled.max(0)
        self.levels = [
            sorted(category_values(train[c]).unique().tolist()) for c in self.schema.categorical
        ]
        self.maps = [{v: i + 1 for i, v in enumerate(levels)} for levels in self.levels]
        self.cardinalities = [len(levels) + 1 for levels in self.levels]
        return self

    def transform(self, frame):
        num = frame[self.schema.numerical].to_numpy(float)
        num = np.where(np.isnan(num), self.fill, num)
        num = self.transformer.transform(num) if self.transformer is not None else num
        cat = [
            category_values(frame[c]).map(m).fillna(0).to_numpy()
            for c, m in zip(self.schema.categorical, self.maps)
        ]
        return np.concatenate([num] + [a[:, None] for a in cat], axis=1).astype("float32")

    def inverse(self, num, cat):
        num = self.inverse_numeric(num)
        values = {c: num[:, i] for i, c in enumerate(self.schema.numerical)}
        for c in self.schema.integer:
            values[c] = values[c].astype("int64")
        for i, (c, levels) in enumerate(zip(self.schema.categorical, self.levels)):
            codes = np.asarray(cat[:, i])
            if (
                not np.equal(codes, np.floor(codes)).all()
                or (codes < 1).any()
                or (codes > len(levels)).any()
            ):
                raise ValueError("Generated unknown/invalid category")
            values[c] = np.asarray(levels)[codes.astype(int) - 1]
        return pd.DataFrame(values, columns=self.schema.columns)

    def inverse_numeric(self, num):
        """The numerical decoder, shared by generation and range snapping."""
        num = np.asarray(num, dtype=float)
        if not np.isfinite(num).all():
            raise ValueError("Nonfinite numerical generation endpoint")
        num = self.transformer.inverse_transform(num) if self.transformer is not None else num
        for i, c in enumerate(self.schema.numerical):
            if c in self.schema.integer:
                num[:, i] = np.rint(num[:, i])
        return num

    def inverse_lookup(self, points=16385):
        """Raw standardized coordinates on a uniform model-space interpolation grid."""
        if not self.schema.numerical:
            return np.empty((points, 0)), np.empty((points, 0))
        lower = self.model_min
        upper = self.model_max
        grid = np.linspace(lower, upper, points)
        raw = self.transformer.inverse_transform(grid)
        return grid.astype("float32"), ((raw - self.mean) / self.scale).astype("float32")
