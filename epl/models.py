"""stage 06: the five models
All models get the same features, folds, inner validation season, budget (N_ITER), seed and objective (log-loss)
"""
import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import log_loss_rows

SPACES = {
    "logreg": {"C": loguniform(1e-3, 100)},
    "rf": {
        "n_estimators": randint(200, 801),
        "max_depth": [3, 5, 8, 12, None],
        "min_samples_leaf": randint(1, 51),
        "max_features": ["sqrt", 0.3, 0.5],
    },
    "xgb": {
        "n_estimators": randint(100, 601),
        "learning_rate": uniform(0.01, 0.19),
        "max_depth": randint(2, 7),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.5, 0.5),
        "min_child_weight": randint(1, 11),
        "reg_lambda": loguniform(0.1, 10),
    },
    "gb": {
        "n_estimators": randint(100, 501),
        "learning_rate": uniform(0.01, 0.19),
        "max_depth": randint(2, 6),
        "subsample": uniform(0.6, 0.4),
        "min_samples_leaf": randint(5, 51),
        "max_features": ["sqrt", None],
    },
    "lstm": {
        "units": [16, 32, 64],
        "dropout": [0.0, 0.2, 0.4],
        "dense": [0, 16, 32],
        "learning_rate": loguniform(1e-4, 3e-3),
        "batch_size": [32, 64, 128],
        "l2": [0.0, 1e-4, 1e-3],
    },
}

NAMES = {"logreg": "Logistic Regression", "rf": "Random Forest", "xgb": "XGBoost",
         "gb": "Gradient Boosting", "lstm": "LSTM"}


def make_model(name, params, seed):
    params = dict(params)
    if name == "logreg":
        return Pipeline([("scale", StandardScaler()),
                         ("lr", LogisticRegression(C=params["C"], max_iter=5000))])
    if name == "rf":
        return RandomForestClassifier(random_state=seed, n_jobs=-1, **params)
    if name == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(objective="multi:softprob", tree_method="hist",
                             eval_metric="mlogloss", random_state=seed, n_jobs=-1, **params)
    if name == "gb":
        return GradientBoostingClassifier(random_state=seed, **params)
    if name == "lstm":
        return LSTMModel(seed=seed, **params)
    raise ValueError(name)


def sample_params(name, n_iter, seed):
    out = []
    for p in ParameterSampler(SPACES[name], n_iter=n_iter, random_state=seed):
        out.append({k: (v.item() if hasattr(v, "item") else v) for k, v in p.items()})
    return out


def tune(name, X_tr, y_tr, X_val, y_val, n_iter, seed, max_epochs=100, patience=10):
    trials = []
    best = (np.inf, None, None)
    for i, params in enumerate(sample_params(name, n_iter, seed)):
        model = make_model(name, params, seed)
        if name == "lstm":
            model.fit(X_tr, y_tr, X_val=X_val, y_val=y_val,
                      max_epochs=max_epochs, patience=patience)
            epochs = model.best_epochs
        else:
            model.fit(X_tr, y_tr)
            epochs = None
        score = float(log_loss_rows(model.predict_proba(X_val), y_val).mean())
        trials.append({"trial": i, "val_log_loss": score, "epochs": epochs, **params})
        if score < best[0]:
            best = (score, params, epochs)
    return best[1], best[0], trials, best[2]


def fit_final(name, params, X, y, seed, epochs=None):
    model = make_model(name, params, seed)
    if name == "lstm":
        model.fit(X, y, epochs=epochs)
    else:
        model.fit(X, y)
    return model


# ------------------------------------------------------------------ LSTM
class LSTMModel:
    """Small Keras LSTM over the last SEQ_LEN feature vectors of both teams.
    Scaling is fitted on the training rows and stored with the model."""

    def __init__(self, units=32, dropout=0.2, dense=16, learning_rate=1e-3,
                 batch_size=64, l2=0.0, seed=42):
        self.params = dict(units=units, dropout=dropout, dense=dense,
                           learning_rate=learning_rate, batch_size=batch_size, l2=l2)
        self.seed = seed
        self.model = None
        self.mu = self.sd = None
        self.best_epochs = None

    def _scale(self, X):
        return ((X - self.mu) / self.sd).astype("float32")

    def _build(self, steps, n_feat):
        import tensorflow as tf
        from tensorflow import keras
        keras.backend.clear_session()   # stops old trial models piling up in memory
        tf.keras.utils.set_random_seed(self.seed)
        p = self.params
        reg = keras.regularizers.l2(p["l2"]) if p["l2"] else None
        inp = keras.Input(shape=(steps, n_feat))
        x = keras.layers.LSTM(p["units"], dropout=p["dropout"], kernel_regularizer=reg)(inp)
        if p["dense"]:
            x = keras.layers.Dense(p["dense"], activation="relu", kernel_regularizer=reg)(x)
        out = keras.layers.Dense(3, activation="softmax")(x)
        m = keras.Model(inp, out)
        m.compile(optimizer=keras.optimizers.Adam(p["learning_rate"]),
                  loss="sparse_categorical_crossentropy")
        return m

    def fit(self, X, y, X_val=None, y_val=None, epochs=None, max_epochs=100, patience=10):
        from tensorflow import keras
        flat = X.reshape(-1, X.shape[-1])
        self.mu, self.sd = flat.mean(axis=0), flat.std(axis=0) + 1e-8
        self.model = self._build(X.shape[1], X.shape[2])
        bs = self.params["batch_size"]
        if X_val is not None:
            stop = keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True)
            h = self.model.fit(self._scale(X), y, validation_data=(self._scale(X_val), y_val),
                               epochs=max_epochs, batch_size=bs, callbacks=[stop], verbose=0)
            self.best_epochs = int(np.argmin(h.history["val_loss"]) + 1)
        else:
            self.model.fit(self._scale(X), y, epochs=int(epochs or 20), batch_size=bs, verbose=0)
            self.best_epochs = int(epochs or 20)
        return self

    def predict_proba(self, X):
        return self.model.predict(self._scale(X), verbose=0)

    # Keras models don't pickle reliably, so save the network as .keras
    def save(self, path):
        self.model.save(str(path) + ".keras")
        np.savez(str(path) + "_meta.npz", mu=self.mu, sd=self.sd, seed=self.seed,
                 best_epochs=self.best_epochs, **{f"p_{k}": v for k, v in self.params.items()})

    @classmethod
    def load(cls, path):
        from tensorflow import keras
        meta = np.load(str(path) + "_meta.npz")
        params = {k[2:]: meta[k].item() for k in meta.files if k.startswith("p_")}
        obj = cls(seed=int(meta["seed"]), **params)
        obj.mu, obj.sd = meta["mu"], meta["sd"]
        obj.best_epochs = int(meta["best_epochs"])
        obj.model = keras.models.load_model(str(path) + ".keras")
        return obj


def build_sequences(ds, team_feats, cols, steps=5):
    base = sorted({c[2:] for c in cols})
    # rows with missing values (warm-up season starts) are never used as history
    tf = team_feats.dropna(subset=base).sort_values(["date", "match_id"]).reset_index(drop=True)
    tf["pos"] = tf.groupby("team").cumcount()
    row_of = {(m, t): i for i, (m, t) in enumerate(zip(tf["match_id"], tf["team"]))}
    rows_by_team = {t: g.index.to_numpy() for t, g in tf.groupby("team")}
    pos = tf["pos"].to_numpy()

    def windows(team_col):
        idx = np.empty((len(ds), steps), dtype=int)
        for n, (mid, team) in enumerate(zip(ds["match_id"], ds[team_col])):
            r = row_of[(mid, team)]
            p = pos[r]
            ps = np.clip(np.arange(p - steps + 1, p + 1), 0, None)
            idx[n] = rows_by_team[team][ps]
        return idx

    win = {"h": windows("home"), "a": windows("away")}
    X = np.empty((len(ds), steps, len(cols)), dtype="float32")
    for j, c in enumerate(cols):
        vals = tf[c[2:]].to_numpy(float)
        X[:, :, j] = vals[win[c[0]]]
    return X
