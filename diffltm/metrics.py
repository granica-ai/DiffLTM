"""Pinned ltm-eval fidelity; publication XGBoost utility and nearest-train DCR."""

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances, r2_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler

FIDELITY_KEYS = ("shape", "trend", "cond_shape_all", "cond_shape_rare")


def subset(frame, n, seed):
    if n > len(frame):
        raise ValueError("Cannot subsample without replacement beyond available rows")
    return frame.sample(n=n, replace=False, random_state=seed)


def fidelity(real, generated, schema, seed, cap=None):
    import ltm_eval

    if ltm_eval.__version__ != "0.1.0":
        raise RuntimeError("Fidelity protocol requires ltm-eval 0.1.0")
    n = min(len(real), len(generated), cap if cap is not None else len(real))
    if n < 1:
        return {k: float("nan") for k in FIDELITY_KEYS}
    real, generated = subset(real, n, seed), subset(generated, n, seed + 1)
    config = ltm_eval.EvalConfig(
        n_bootstrap=0,
        seed=seed,
        n_jobs=1,
        split_holdout_ideal=False,
        max_eval_n=None,
        clean="safe",
        metric_kwargs={
            "conditional_fidelity": {"rare_threshold": 0.05, "min_count": 10, "num_bins": 10}
        },
    )
    report = ltm_eval.evaluate(
        real[schema.columns],
        generated[schema.columns],
        schema=ltm_eval.Schema(numeric=schema.numerical, categorical=schema.categorical),
        config=config,
        metrics=["fidelity", "conditional_fidelity"],
    )
    stats = {
        k: float(s.value)
        for result in report.results
        for k, s in result.stats.items()
        if k in FIDELITY_KEYS
    }
    if set(stats) != set(FIDELITY_KEYS):
        raise RuntimeError("Unexpected ltm-eval metric API")
    return stats


def dcr(train, holdout, generated, schema, seed):
    frames = [subset(f, min(10000, len(f)), seed).dropna() for f in [train, holdout, generated]]
    if len(frames[0]) < 2 or any(len(f) == 0 for f in frames[1:]):
        return float("nan")
    num, cat = schema.numerical, schema.categorical
    ranges = frames[0][num].max() - frames[0][num].min()
    ranges = ranges.replace(0, 1)
    encoder = (
        OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(frames[0][cat])
        if cat
        else None
    )

    def features(frame):
        parts = [frame[num].to_numpy(float) / ranges.to_numpy()]
        if encoder is not None:
            parts.append(encoder.transform(frame[cat]))
        return np.concatenate(parts, 1).astype("float32")

    reference, real, syn = map(features, frames)

    def distances(query):
        return np.concatenate(
            [
                pairwise_distances(query[i : i + 1024], reference, n_jobs=1).min(1)
                for i in range(0, len(query), 1024)
            ]
        )

    denominator = float(np.median(distances(real)))
    return float(np.median(distances(syn))) / denominator if denominator > 0 else float("inf")


def mle(
    train,
    holdout,
    generated,
    schema,
    seed,
    rounds=10,
    fit_n=None,
):
    """Ratio of mean TSTR/TRTR scores; unsupported fits remain undefined."""
    from xgboost import XGBClassifier, XGBRegressor

    n = len(holdout) if fit_n is None else fit_n
    if min(len(train), len(generated)) < n:
        raise ValueError("MLE pools smaller than matched requested fit size")
    num = [c for c in schema.numerical if c != schema.target]
    cat = [c for c in schema.categorical if c != schema.target]
    combined = train  # Publication bootstrap uses real-train-fitted encoders/scaler.
    enc = (
        OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(combined[cat])
        if cat
        else None
    )
    label = LabelEncoder().fit(train[schema.target]) if schema.task != "regression" else None

    def unscaled(frame):
        parts = []
        if num:
            parts.append(frame[num].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(float))
        if cat:
            parts.append(enc.transform(frame[cat]))
        return np.concatenate(parts, 1)

    scaler = StandardScaler().fit(unscaled(combined))

    def xy(frame):
        x = np.nan_to_num(scaler.transform(unscaled(frame)), nan=0.0, posinf=0.0, neginf=0.0)
        y = (
            label.transform(frame[schema.target])
            if label is not None
            else frame[schema.target].to_numpy(float)
        )
        return x, y

    x_test, y_test = xy(holdout)
    scores = {"trtr": [], "tstr": []}
    for key, pool in [("trtr", train), ("tstr", generated)]:
        rng = np.random.default_rng(seed)
        x_pool, y_pool = xy(pool)
        for _ in range(rounds):
            indices = rng.choice(len(pool), n, replace=False)
            x, y = x_pool[indices], y_pool[indices]
            if label is not None and not np.array_equal(
                np.unique(y), np.arange(len(label.classes_))
            ):
                scores[key].append(float("nan"))
                continue
            cls = XGBRegressor if label is None else XGBClassifier
            estimator = cls(
                n_estimators=100, max_depth=10, learning_rate=0.1, random_state=42, n_jobs=1
            )
            estimator.fit(x, y)
            if label is None:
                score = r2_score(y_test, estimator.predict(x_test), force_finite=False)
            elif len(np.unique(y_test)) != len(label.classes_):
                score = float("nan")
            else:
                probs = estimator.predict_proba(x_test)
                score = (
                    roc_auc_score(y_test, probs[:, 1])
                    if len(label.classes_) == 2
                    else roc_auc_score(y_test, probs, multi_class="ovr")
                )
            scores[key].append(float(score))
    real, syn = np.mean(scores["trtr"]), np.mean(scores["tstr"])
    ratio = float(syn / real) if np.isfinite([real, syn]).all() and real != 0 else float("nan")
    return ratio, scores
