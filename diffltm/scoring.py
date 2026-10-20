"""Final unconditional scoring from generated and reference tables."""

import numpy as np

from .artifacts import request_seed
from .metrics import dcr, fidelity, mle, subset

METRIC_SETTINGS = {
    "fidelity_backend": "ltm-eval==0.1.0",
    "clean": "safe",
    "num_bins": 10,
    "min_count": 10,
    "rare_threshold": 0.05,
    "mle_rounds": 10,
    "xgb_trees": 100,
    "xgb_depth": 10,
    "xgb_learning_rate": 0.1,
    "xgb_seed": 42,
    "dcr_cap": 10000,
}


def score_rows(scores, result_identity, **cell):
    return [
        dict(
            **result_identity,
            **cell,
            metric=metric,
            value=float(value) if np.isfinite(value) else float("nan"),
            status="complete" if np.isfinite(value) else "undefined",
            omission_reason=None
            if np.isfinite(value)
            else {
                "cond_shape_rare": "No eligible rare-group aggregate after safe cleaning (count >= 10, frequency < .05)",
                "cond_shape_all": "No eligible conditional aggregate after safe cleaning",
                "trend": "No defined column-pair aggregate after safe cleaning",
                "shape": "No defined column aggregate after safe cleaning",
                "mle": "Undefined fit/score or zero TRTR denominator; see mle_rounds",
                "dcr": "Insufficient complete rows or zero median holdout distance",
            }.get(metric, "Metric is not applicable to the scored population"),
        )
        for metric, value in scores.items()
    ]


def unconditional(train, test, samples, schema, result_identity, generation_seed):
    seed = request_seed(result_identity["training_seed"], "unconditional", "scoring")
    scores = fidelity(test, subset(samples, len(test), seed), schema, seed)
    scores["mle"], rounds = mle(
        train,
        test,
        samples,
        schema,
        seed,
    )
    scores["dcr"] = dcr(train, test, samples, schema, seed)
    rows = score_rows(
        scores,
        result_identity,
        task="unconditional",
        method="unconditional",
        slice="all",
        scoring_seed=seed,
        generation_seed=generation_seed,
        requested=len(train),
        returned=len(samples),
        n_reference=len(test),
        n_scored=len(test),
    )
    for row in rows:
        if row["metric"] == "dcr":
            row.update(
                n_scored=min(10000, len(samples)),
                n_reference=min(10000, len(test)),
                n_train_reference=min(10000, len(train)),
            )
        elif row["metric"] == "mle":
            row.update(n_fit_per_round=len(test), fit_rounds=10)
    return rows, dict(mle_rounds=rounds)
