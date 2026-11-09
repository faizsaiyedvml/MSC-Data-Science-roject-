"""
stage 11 : ablation
"""
import pickle

import numpy as np
import pandas as pd

from . import metrics as M
from .features import CATEGORIES, feature_category
from .models import fit_final, tune


def variants(cols):
    by_cat = {c: [x for x in cols if feature_category(x) == c] for c in CATEGORIES}
    v = {"full": list(cols)}
    for c in CATEGORIES:
        v[f"minus_{c}"] = [x for x in cols if feature_category(x) != c]
    v["conventional_only"] = by_cat["conventional"]
    for c in ["elo", "xg", "form"]:
        v[f"conventional+{c}"] = by_cat["conventional"] + by_cat[c]
    return v


REFERENCE = {**{f"minus_{c}": "full" for c in CATEGORIES},
             **{f"conventional+{c}": "conventional_only" for c in ["elo", "xg", "form"]}}


def run_ablation(name, data, folds, bench, cfg, P, tracker, log=print):
    ck = P["checkpoints"] / f"ablation_{name}.pkl"
    if ck.exists():
        log(f"  {name}: ablation checkpoint found")
        return pickle.load(open(ck, "rb"))
    info = {f["fold"]: f for f in bench["folds"]}
    rows = []
    for vname, vcols in variants(data.cols).items():
        for f in folds:
            if vname == "full":
                # identical to the benchmark run, so reuse its predictions
                p = bench["preds"][bench["preds"]["fold"] == f.k]
                proba, y = p[["p_H", "p_D", "p_A"]].to_numpy(), p["y"].to_numpy()
            else:
                params, epochs = info[f.k]["params"], info[f.k]["epochs"]
                if cfg.ABLATION_RETUNE:
                    Xi, yi, _ = data.get(name, f.inner_train, vcols)
                    Xv, yv, _ = data.get(name, [f.inner_val], vcols)
                    params, _, _, epochs = tune(name, Xi, yi, Xv, yv, cfg.N_ITER, cfg.SEED,
                                                cfg.LSTM_MAX_EPOCHS, cfg.LSTM_PATIENCE)
                Xtr, ytr, _ = data.get(name, f.train_seasons, vcols)
                Xte, y, _ = data.get(name, [f.test_season], vcols)
                proba = fit_final(name, params, Xtr, ytr, cfg.SEED, epochs).predict_proba(Xte)
            res = M.summary(proba, y)
            pcf1 = M.per_class(proba, y).set_index("class")["f1"]
            rows.append({"model": name, "variant": vname, "fold": f.k, "season": f.test_season,
                         "n_features": len(vcols), **res,
                         **{f"f1_{c}": pcf1[c] for c in M.CLASSES}})
            tracker.log(f"abl_{name}_{vname}_fold{f.k}", metrics=res,
                        tags={"stage": "ablation", "model": name, "variant": vname})
        log(f"  {name} {vname}: done")
    out = pd.DataFrame(rows)
    pickle.dump(out, open(ck, "wb"))
    return out


def summarise(abl):
    """per-fold deltas against the reference variant, then mean and SD.
    psitive delta_rps / delta_log_loss = worse; negative delta_accuracy / f1 = worse."""
    metrics = ["rps", "log_loss", "accuracy", "macro_f1", "f1_H", "f1_D", "f1_A"]
    out = []
    for (model, var), g in abl.groupby(["model", "variant"]):
        if var not in REFERENCE:
            continue
        ref = abl[(abl["model"] == model) & (abl["variant"] == REFERENCE[var])].set_index("fold")
        g = g.set_index("fold")
        d = g[metrics] - ref.loc[g.index, metrics]
        row = {"model": model, "variant": var, "reference": REFERENCE[var],
               "n_features": int(g["n_features"].iloc[0])}
        for m in metrics:
            row[f"delta_{m}_mean"] = d[m].mean()
            row[f"delta_{m}_sd"] = d[m].std()
        out.append(row)
    return pd.DataFrame(out)
