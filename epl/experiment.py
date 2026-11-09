"""Stage 08: benchmark runner (E1) and per-class results (E3)"""
import pickle
import time

import joblib
import numpy as np
import pandas as pd

from . import metrics as M
from .models import LSTMModel, build_sequences, fit_final, tune
from .validation import label


class Data:
    """holds the dataset and hands out X/y for any season set and column subset."""

    def __init__(self, ds, team_feats, cols, steps=5, need_seq=True):
        self.ds = ds.reset_index(drop=True)
        self.cols = list(cols)
        self.X = self.ds[self.cols].to_numpy(float)
        self.y = self.ds["y"].to_numpy(int)
        self.season = self.ds["season"].to_numpy()
        self.seq = build_sequences(self.ds, team_feats, self.cols, steps) if need_seq else None

    def get(self, model, seasons, cols=None):
        mask = np.isin(self.season, seasons)
        j = [self.cols.index(c) for c in (cols or self.cols)]
        X = self.seq[mask][:, :, j] if model == "lstm" else self.X[mask][:, j]
        return X, self.y[mask], np.where(mask)[0]


def model_path(models_dir, name, k):
    return models_dir / f"{name}_fold{k}"


def save_model(model, name, path):
    if name == "lstm":
        model.save(path)
    else:
        joblib.dump(model, str(path) + ".joblib")


def load_model(name, path):
    return LSTMModel.load(path) if name == "lstm" else joblib.load(str(path) + ".joblib")


def run_model(name, data, folds, cfg, P, tracker, log=print):
    ck = P["checkpoints"] / f"bench_{name}.pkl"
    if ck.exists():
        log(f"  {name}: checkpoint found, skipping")
        return pickle.load(open(ck, "rb"))

    preds, fold_info, trials_all = [], [], []
    for f in folds:
        t0 = time.time()
        Xi, yi, _ = data.get(name, f.inner_train)
        Xv, yv, _ = data.get(name, [f.inner_val])
        params, val_ll, trials, epochs = tune(name, Xi, yi, Xv, yv, cfg.N_ITER, cfg.SEED,
                                              cfg.LSTM_MAX_EPOCHS, cfg.LSTM_PATIENCE)
        Xtr, ytr, _ = data.get(name, f.train_seasons)
        Xte, yte, idx = data.get(name, [f.test_season])
        model = fit_final(name, params, Xtr, ytr, cfg.SEED, epochs)
        proba = model.predict_proba(Xte)
        save_model(model, name, model_path(P["models"], name, f.k))

        p = pd.DataFrame(proba, columns=["p_H", "p_D", "p_A"])
        p["match_id"] = data.ds.loc[idx, "match_id"].values
        p["season"] = f.test_season
        p["fold"] = f.k
        p["model"] = name
        p["y"] = yte
        preds.append(p)

        res = M.summary(proba, yte)
        fold_info.append({"fold": f.k, "test_season": f.test_season, "params": params,
                          "epochs": epochs, "val_log_loss": val_ll, **res})
        for t in trials:
            trials_all.append({"model": name, "fold": f.k, **t})
        tracker.log(f"bench_{name}_fold{f.k}", params={**params, "epochs": epochs},
                    metrics={**res, "val_log_loss": val_ll},
                    tags={"stage": "benchmark", "model": name, "test_season": label(f.test_season)})
        log(f"  {name} fold {f.k} ({label(f.test_season)}): rps={res['rps']:.4f} "
            f"acc={res['accuracy']:.3f}  [{time.time() - t0:.0f}s]")

    out = {"model": name, "preds": pd.concat(preds, ignore_index=True),
           "folds": fold_info, "trials": pd.DataFrame(trials_all)}
    pickle.dump(out, open(ck, "wb"))
    return out


def load_benchmarks(P, models):
    out = {}
    for m in models:
        f = P["checkpoints"] / f"bench_{m}.pkl"
        if f.exists():
            out[m] = pickle.load(open(f, "rb"))
    return out


def results_tables(bench):
    preds = pd.concat([b["preds"] for b in bench.values()], ignore_index=True)
    pr = preds[["p_H", "p_D", "p_A"]].to_numpy()
    preds["pred"] = pr.argmax(axis=1)
    preds["correct"] = (preds["pred"] == preds["y"]).astype(int)
    preds["rps"] = M.rps(pr, preds["y"])
    preds["log_loss"] = M.log_loss_rows(pr, preds["y"])

    season_rows, pc_rows, cm_rows = [], [], []
    for (m, s), g in preds.groupby(["model", "season"]):
        p = g[["p_H", "p_D", "p_A"]].to_numpy()
        season_rows.append({"model": m, "season": s, "n": len(g), **M.summary(p, g["y"])})
        pc = M.per_class(p, g["y"]); pc["model"], pc["season"] = m, s
        pc_rows.append(pc)
    overall_rows = []
    for m, g in preds.groupby("model"):
        p = g[["p_H", "p_D", "p_A"]].to_numpy()
        overall_rows.append({"model": m, "n": len(g), **M.summary(p, g["y"])})
        pc = M.per_class(p, g["y"]); pc["model"], pc["season"] = m, "pooled"
        pc_rows.append(pc)
        cm = M.confusion(p, g["y"]).reset_index().rename(columns={"index": "actual"})
        cm["model"] = m
        cm_rows.append(cm)

    season = pd.DataFrame(season_rows)
    overall = pd.DataFrame(overall_rows)
    sd = season.groupby("model")[["accuracy", "macro_f1", "log_loss", "rps"]].std().add_suffix("_sd")
    mean = season.groupby("model")[["accuracy", "macro_f1", "log_loss", "rps"]].mean().add_suffix("_season_mean")
    overall = overall.set_index("model").join(mean).join(sd).reset_index().sort_values("rps")
    per_class = pd.concat(pc_rows, ignore_index=True)
    pooled = per_class[per_class["season"] == "pooled"]
    hardest = pooled.loc[pooled.groupby("model")["f1"].idxmin(), ["model", "class", "f1"]]
    class_mean = pooled.groupby("class")["f1"].mean().reindex(M.CLASSES).reset_index()
    return {"predictions": preds, "season": season, "overall": overall,
            "per_class": per_class, "confusion": pd.concat(cm_rows, ignore_index=True),
            "hardest_by_model": hardest, "class_f1_mean": class_mean}
