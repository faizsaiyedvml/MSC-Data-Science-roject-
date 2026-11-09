"""Stage 10 : SHAP for the two best models by mean RPS
"""
import pickle

import numpy as np
import pandas as pd

from .experiment import load_model, model_path
from .features import feature_category
from .metrics import CLASSES

UNITS = {"rf": "probability", "xgb": "log-odds", "gb": "log-odds",
         "logreg": "log-odds", "lstm": "probability"}


def _to_ncf(vals, n_classes=3):
    """normalise SHAP output to shape --- n, classes, features"""
    if isinstance(vals, list):
        return np.stack(vals, axis=1)
    vals = np.asarray(vals)
    if vals.ndim == 3 and vals.shape[2] == n_classes:
        return vals.transpose(0, 2, 1)
    return vals


def shap_xgb(model, X):
    import xgboost as xgb
    c = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True, strict_shape=True)
    return c[:, :, :-1], c[:, :, -1]


def shap_gb(model, X):
    """treeSHAP on each class's regression trees, scaled by the learning rate."""
    import shap
    n_est, n_cls = model.estimators_.shape
    vals = np.zeros((len(X), n_cls, X.shape[1]))
    base = model._raw_predict_init(X).astype(float).copy()
    for k in range(n_cls):
        for t in range(n_est):
            tree = model.estimators_[t, k]
            e = shap.TreeExplainer(tree)
            vals[:, k, :] += model.learning_rate * e.shap_values(X)
            base[:, k] += model.learning_rate * float(np.ravel(e.expected_value)[0])
    return vals, base


def shap_rf(model, X):
    import shap
    e = shap.TreeExplainer(model)
    return _to_ncf(e.shap_values(X)), np.ravel(e.expected_value)


def shap_logreg(model, X, X_bg):
    scaler, lr = model.named_steps["scale"], model.named_steps["lr"]
    Xs, Bs = scaler.transform(X), scaler.transform(X_bg)
    try:
        import shap
        e = shap.LinearExplainer((lr.coef_, lr.intercept_), shap.maskers.Independent(Bs, max_samples=len(Bs)))
        vals = _to_ncf(e.shap_values(Xs))
    except Exception:
        # same quantity LinearExplainer gives with independent masking
        vals = (Xs - Bs.mean(axis=0))[:, None, :] * lr.coef_[None, :, :]
    return vals, None


def expected_gradients(f, X, bg, n_samples=200, seed=0):
    """expected gradients (erion et al., 2021). f maps a tf tensor (n, ...) to (n, classes).
    Returns (n, classes, *feature_shape)."""
    import tensorflow as tf
    rng = np.random.default_rng(seed)
    out = None
    for i in range(len(X)):
        x = np.repeat(X[i:i + 1], n_samples, axis=0)
        b = bg[rng.integers(0, len(bg), n_samples)]
        a = rng.uniform(size=(n_samples,) + (1,) * (X.ndim - 1))
        pts = tf.convert_to_tensor((b + a * (x - b)).astype("float32"))
        rows = []
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(pts)
            y = f(pts)
            outs = [y[:, k] for k in range(y.shape[1])]
        for k, yk in enumerate(outs):
            g = tape.gradient(yk, pts).numpy()
            rows.append((g * (x - b)).mean(axis=0))
        del tape
        r = np.stack(rows)
        if out is None:
            out = np.zeros((len(X),) + r.shape)
        out[i] = r
    return out


def shap_lstm(model, X, X_bg, n_bg=200, seed=0):
    rng = np.random.default_rng(seed)
    Xs = model._scale(X)
    Bs = model._scale(X_bg[rng.choice(len(X_bg), min(n_bg, len(X_bg)), replace=False)])
    try:
        import shap
        e = shap.GradientExplainer(model.model, Bs)
        v = e.shap_values(Xs)
        v = np.stack(v, axis=-1) if isinstance(v, list) else np.asarray(v)  # (n, t, f, c)
        vals = v.sum(axis=1).transpose(0, 2, 1)
    except Exception:
        v = expected_gradients(lambda t: model.model(t, training=False), Xs, Bs, seed=seed)
        vals = v.sum(axis=2)                                                  # (n, c, f)
    return vals, None


def explain_fold(name, model, X_test, X_train):
    if name == "xgb":
        return shap_xgb(model, X_test)[0]
    if name == "gb":
        return shap_gb(model, X_test)[0]
    if name == "rf":
        return shap_rf(model, X_test)[0]
    if name == "logreg":
        return shap_logreg(model, X_test, X_train)[0]
    if name == "lstm":
        return shap_lstm(model, X_test, X_train)[0]
    raise ValueError(name)


def run_shap(name, data, folds, P, log=print):
    ck = P["checkpoints"] / f"shap_{name}.pkl"
    if ck.exists():
        log(f"  {name}: SHAP checkpoint found")
        return pickle.load(open(ck, "rb"))
    vals, Xs, ids, seasons = [], [], [], []
    for f in folds:
        model = load_model(name, model_path(P["models"], name, f.k))
        Xte, _, idx = data.get(name, [f.test_season])
        Xtr, _, _ = data.get(name, f.train_seasons)
        v = explain_fold(name, model, Xte, Xtr)
        vals.append(v)
        Xs.append(data.X[idx])                       # tabular values for plots
        ids.append(data.ds.loc[idx, "match_id"].values)
        seasons.append(np.full(len(idx), f.test_season))
        log(f"  {name} fold {f.k}: SHAP done ({len(idx)} matches)")
    out = {"model": name, "values": np.concatenate(vals), "X": np.concatenate(Xs),
           "match_id": np.concatenate(ids), "season": np.concatenate(seasons),
           "cols": data.cols, "units": UNITS[name]}
    pickle.dump(out, open(ck, "wb"))
    return out


def shap_tables(res):
    local, glob, cat = [], [], []
    for name, r in res.items():
        v, cols = r["values"], r["cols"]
        n, c, f = v.shape
        local.append(pd.DataFrame({
            "match_id": np.repeat(r["match_id"], c * f),
            "season": np.repeat(r["season"], c * f),
            "model": name,
            "class": np.tile(np.repeat(CLASSES, f), n),
            "feature": np.tile(cols, n * c),
            "feature_value": np.repeat(r["X"], c, axis=0).ravel(),
            "shap": v.ravel(),
            "units": r["units"],
        }))
        mabs = np.abs(v).mean(axis=0)                 # (classes, features)
        for k, cl in enumerate(CLASSES + ["all"]):
            row = mabs.mean(axis=0) if cl == "all" else mabs[k]
            g = pd.DataFrame({"model": name, "class": cl, "feature": cols,
                              "category": [feature_category(x) for x in cols],
                              "mean_abs_shap": row})
            g["rank"] = g["mean_abs_shap"].rank(ascending=False, method="min").astype(int)
            glob.append(g)
            cg = g.groupby("category")["mean_abs_shap"].agg(["sum", "mean", "count"]).reset_index()
            cg.columns = ["category", "total_mean_abs_shap", "mean_per_feature", "n_features"]
            cg["share_of_total"] = cg["total_mean_abs_shap"] / cg["total_mean_abs_shap"].sum()
            cg["model"], cg["class"] = name, cl
            cat.append(cg)
    return (pd.concat(local, ignore_index=True), pd.concat(glob, ignore_index=True),
            pd.concat(cat, ignore_index=True))


def beeswarm_plots(res, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap
    for name, r in res.items():
        for k, cl in enumerate(CLASSES):
            plt.figure()
            shap.summary_plot(r["values"][:, k, :], pd.DataFrame(r["X"], columns=r["cols"]),
                              show=False, max_display=20)
            plt.title(f"{name}: SHAP for P({cl}) [{r['units']}]")
            plt.tight_layout()
            plt.savefig(fig_dir / f"shap_beeswarm_{name}_{cl}.png", dpi=150)
            plt.close("all")
