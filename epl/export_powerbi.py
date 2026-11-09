"""Stage 12: flat CSV tables for the Power BI dashboard"""
import shutil

import pandas as pd

from .metrics import CLASSES
from .models import NAMES


def export(P, ds, dictionary, tables, log=print):
    out = P["powerbi"]
    t = P["tables"]

    dim_match = ds[["match_id", "season", "date", "home", "away", "fthg", "ftag", "ftr"]].copy()
    dim_match["season_label"] = dim_match["season"].map(lambda s: f"{s}/{(s + 1) % 100:02d}")
    dim_match.to_csv(out / "dim_match.csv", index=False)

    dim_model = pd.DataFrame({"model": list(NAMES), "model_name": list(NAMES.values())})
    dim_model.to_csv(out / "dim_model.csv", index=False)

    dictionary.to_csv(out / "dim_feature.csv", index=False)

    if "predictions" in tables:
        f = tables["predictions"].copy()
        f["predicted"] = f["pred"].map(dict(enumerate(CLASSES)))
        f["actual"] = f["y"].map(dict(enumerate(CLASSES)))
        f[["match_id", "model", "season", "fold", "p_H", "p_D", "p_A", "predicted", "actual",
           "correct", "rps", "log_loss"]].to_csv(out / "fact_predictions.csv", index=False)

    # copy the results tables across as they are
    for name in ["model_metrics_season", "model_metrics_overall", "per_class_metrics",
                 "confusion_matrices", "stat_tests_friedman", "stat_tests_ranks",
                 "stat_tests_pairwise", "shap_local", "shap_global", "shap_category",
                 "ablation_summary", "ablation_by_fold"]:
        src = t / f"{name}.csv"
        if src.exists():
            shutil.copy(src, out / f"{name}.csv")
    log(f"  Power BI files written to {out}")
