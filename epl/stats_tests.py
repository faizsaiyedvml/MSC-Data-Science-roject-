"""stage 09 : Friedman, Nemenyi critical difference and pairwise
Wilcoxon signed-rank tests with Holm correction (Demsar, 2006).
Blocks are the test seasons."""
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

from .metrics import LOWER_IS_BETTER

# Nemenyi q at alpha = 0.05 (Demsar 2006, Table 5a), indexed by number of models
Q_05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031}


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for i, j in enumerate(order):
        running = max(running, (m - i) * p[j])
        adj[j] = min(1.0, running)
    return adj


def average_ranks(table, lower_better):
    vals = table.to_numpy(float)
    if not lower_better:
        vals = -vals
    ranks = np.apply_along_axis(rankdata, 1, vals)
    return pd.Series(ranks.mean(axis=0), index=table.columns)


def nemenyi_cd(k, n, q=None):
    q = q or Q_05[k]
    return q * np.sqrt(k * (k + 1) / (6.0 * n))


def run_tests(season_metrics, metric_names=("rps", "log_loss", "accuracy", "macro_f1")):
    friedman, ranks, pairs = [], [], []
    for metric in metric_names:
        t = season_metrics.pivot(index="season", columns="model", values=metric).dropna()
        k, n = t.shape[1], t.shape[0]
        lb = LOWER_IS_BETTER[metric]
        r = average_ranks(t, lb)
        cd = nemenyi_cd(k, n) if k in Q_05 else np.nan
        if k >= 3:
            stat, p = friedmanchisquare(*[t[c] for c in t.columns])
        else:
            stat, p = np.nan, np.nan
        friedman.append({"metric": metric, "k_models": k, "n_seasons": n,
                         "friedman_chi2": stat, "p_value": p, "nemenyi_cd": cd})
        for m in t.columns:
            ranks.append({"metric": metric, "model": m, "avg_rank": r[m]})

        rows = []
        for a, b in combinations(t.columns, 2):
            d = t[a] - t[b]
            if np.allclose(d, 0):
                w, p = np.nan, 1.0
            else:
                with warnings.catch_warnings():
                    # n = 8 seasons; scipy warns when ties force the normal approximation
                    warnings.simplefilter("ignore", UserWarning)
                    w, p = wilcoxon(t[a], t[b])
            rows.append({"metric": metric, "model_a": a, "model_b": b,
                         "mean_diff": d.mean(), "wilcoxon_W": w, "p_raw": p,
                         "rank_diff": abs(r[a] - r[b]),
                         "nemenyi_sig": abs(r[a] - r[b]) > cd if not np.isnan(cd) else None})
        if rows:
            adj = holm([x["p_raw"] for x in rows])
            for x, pa in zip(rows, adj):
                x["p_holm"] = pa
                x["holm_sig_05"] = pa < 0.05
        pairs.extend(rows)
    return pd.DataFrame(friedman), pd.DataFrame(ranks), pd.DataFrame(pairs)
