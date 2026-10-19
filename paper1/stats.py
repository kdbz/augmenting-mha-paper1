"""Paired comparisons over repeated cross-validation runs.

Runs from repeated CV share training data, so an ordinary paired t-test is too confident.
The Nadeau-Bengio correction inflates the variance by n_test / n_train. The confidence interval uses
the same corrected variance, so "CI excludes 0" and "p < .05" always agree. A plain bootstrap over
the runs would ignore the overlap and give an interval that is too narrow.
"""
import numpy as np
from scipy import stats


def nadeau_bengio(d, n_train, n_test):
    d = np.asarray(d, float)
    k, v = len(d), d.var(ddof=1)
    if v == 0:
        return 0.0, 1.0
    t = d.mean() / np.sqrt(v * (1 / k + n_test / n_train))
    return float(t), float(2 * stats.t.sf(abs(t), k - 1))


def corrected_ci(d, n_train, n_test, level=0.95):
    d = np.asarray(d, float)
    k = len(d)
    half = stats.t.ppf(0.5 + level / 2, k - 1) * np.sqrt(d.var(ddof=1) * (1 / k + n_test / n_train))
    return float(d.mean() - half), float(d.mean() + half)


def compare(res, a, b, n_train, n_test):
    """Positive delta means `a` has the lower (better) MAE."""
    d = np.array(res[b]["mae"]) - np.array(res[a]["mae"])
    t, p = nadeau_bengio(d, n_train, n_test)
    lo, hi = corrected_ci(d, n_train, n_test)
    return {"delta_mae": float(d.mean()), "ci95": [lo, hi], "t": t, "p": p, "n_runs": int(len(d)),
            "a_wins": int((d > 0).sum())}
