"""stage 04: featureselection
"""
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

from .features import feature_category

DERIVED_FIRST = ["form_gd"]


def _r2_on_rest(X, j):
    y = X[:, j]
    A = np.column_stack([np.ones(len(X)), np.delete(X, j, axis=1)])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    tss = ((y - y.mean()) ** 2).sum()
    return 1.0 if tss == 0 else 1 - (resid ** 2).sum() / tss


def drop_exact_collinear(df, cols, tol=1e-9):
    cols = list(cols)
    order = sorted(cols, key=lambda c: (c[2:] not in DERIVED_FIRST, cols.index(c)))
    log = []
    for c in order:
        if c not in cols or len(cols) < 2:
            continue
        X = df[cols].to_numpy(float)
        if _r2_on_rest(X, cols.index(c)) > 1 - tol:
            cols.remove(c)
            log.append({"feature": c, "vif": np.inf, "reason": "exact linear combination"})
    return cols, log


def compute_vif(df, cols):
    X = np.column_stack([np.ones(len(df)), df[cols].to_numpy(float)])
    return pd.Series([variance_inflation_factor(X, i + 1) for i in range(len(cols))], index=cols)


def vif_screen(df, cols, threshold=10.0):
    cols, log = drop_exact_collinear(df, cols)
    while True:
        vif = compute_vif(df, cols)
        cats = pd.Series({c: feature_category(c) for c in cols})
        counts = cats.value_counts()
        allowed = [c for c in cols if counts[cats[c]] > 1]
        cand = vif[allowed]
        if cand.empty or cand.max() <= threshold:
            break
        worst = cand.idxmax()
        log.append({"feature": worst, "vif": float(cand.max()), "reason": f"VIF > {threshold}"})
        cols.remove(worst)
    final = compute_vif(df, cols)
    return cols, pd.DataFrame(log), final
