"""stage 07: evaluation metrics. class order is home, draw, Aaway"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

CLASSES = ["H", "D", "A"]
EPS = 1e-15


def rps(proba, y):
    """Ranked probability score per match (lower is better)"""
    proba = np.asarray(proba, float)
    obs = np.eye(3)[np.asarray(y, int)]
    cp, co = np.cumsum(proba, axis=1)[:, :2], np.cumsum(obs, axis=1)[:, :2]
    return 0.5 * ((cp - co) ** 2).sum(axis=1)


def log_loss_rows(proba, y):
    p = np.clip(np.asarray(proba, float)[np.arange(len(y)), np.asarray(y, int)], EPS, 1)
    return -np.log(p)


def summary(proba, y):
    y = np.asarray(y, int)
    pred = np.asarray(proba).argmax(axis=1)
    return {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0),
        "log_loss": float(log_loss_rows(proba, y).mean()),
        "rps": float(rps(proba, y).mean()),
    }


def per_class(proba, y):
    pred = np.asarray(proba).argmax(axis=1)
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], zero_division=0)
    return pd.DataFrame({"class": CLASSES, "precision": p, "recall": r, "f1": f, "support": s})


def confusion(proba, y):
    pred = np.asarray(proba).argmax(axis=1)
    cm = confusion_matrix(y, pred, labels=[0, 1, 2])
    return pd.DataFrame(cm, index=[f"true_{c}" for c in CLASSES],
                        columns=[f"pred_{c}" for c in CLASSES])


LOWER_IS_BETTER = {"accuracy": False, "macro_f1": False, "log_loss": True, "rps": True}
