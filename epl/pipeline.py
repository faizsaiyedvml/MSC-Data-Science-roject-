import json
import types

import pandas as pd

from . import config
from . import ablation as AB
from . import data_collection as DC
from . import data_merge as DM
from . import experiment as EX
from . import explain as XP
from . import export_powerbi as PB
from . import feature_selection as FS
from . import features as FE
from . import stats_tests as ST
from .tracking import Tracker
from .validation import make_folds


def settings(**overrides):
    keys = ["SEED", "N_ITER", "MIN_TRAIN_SEASONS", "WARMUP_SEASON", "STUDY_SEASONS",
            "SEQ_LEN", "VIF_THRESHOLD", "DATE_TOLERANCE_DAYS", "MODELS",
            "ELO_BURNIN_SEASONS", "ELO_K", "ELO_HOME_ADV", "ELO_START",
            "LSTM_MAX_EPOCHS", "LSTM_PATIENCE", "ABLATION_RETUNE", "OUT_DIR"]
    s = {k: getattr(config, k) for k in keys}
    s.update({k: v for k, v in overrides.items() if v is not None})
    return types.SimpleNamespace(**s)


class Pipeline:
    def __init__(self, cfg=None, log=print):
        self.cfg = cfg or settings()
        self.P = config.paths(self.cfg.OUT_DIR)
        self.log = log
        self._tracker = None
        self._data = None

    @property
    def tracker(self):
        if self._tracker is None:
            self._tracker = Tracker(self.P["root"])
        return self._tracker

    @property
    def folds(self):
        return make_folds(self.cfg.STUDY_SEASONS, self.cfg.MIN_TRAIN_SEASONS)

    def data(self):
        if self._data is None:
            ds = pd.read_csv(self.P["processed"] / "epl_dataset.csv", parse_dates=["date"])
            tf = pd.read_csv(self.P["processed"] / "team_features.csv", parse_dates=["date"])
            cols = json.load(open(self.P["processed"] / "selected_features.json"))["features"]
            need_seq = "lstm" in self.cfg.MODELS
            self._data = EX.Data(ds, tf, cols, self.cfg.SEQ_LEN, need_seq)
        return self._data

    # 01 ------------------------------------------------------------------
    def s01_collect(self):
        seasons = [self.cfg.WARMUP_SEASON] + list(self.cfg.STUDY_SEASONS)
        DC.collect_all(seasons, self.elo_burnin(), self.P["raw"], self.log)

    def elo_burnin(self):
        w = self.cfg.WARMUP_SEASON
        return list(range(w - self.cfg.ELO_BURNIN_SEASONS, w))

    # 02 ------------------------------------------------------------------
    def s02_merge(self, synthetic=False):
        if synthetic:
            from .synthetic import make_league
            m = make_league(self.cfg.WARMUP_SEASON, max(self.cfg.STUDY_SEASONS), seed=self.cfg.SEED)
            qa = {"source": "synthetic", "rows_final": len(m)}
        else:
            raw = self.P["raw"]
            fd = pd.read_csv(raw / "fd_all.csv", low_memory=False)
            us = pd.read_csv(raw / "understat_all.csv", parse_dates=["date"])
            burn = pd.read_csv(raw / "fd_elo_burnin.csv", low_memory=False)
            m, qa = DM.merge_sources(fd, us, burn, self.cfg.DATE_TOLERANCE_DAYS,
                                     elo_k=self.cfg.ELO_K, elo_home_adv=self.cfg.ELO_HOME_ADV,
                                     elo_start=self.cfg.ELO_START)
        m.to_csv(self.P["processed"] / "merged_matches.csv", index=False)
        DM.save_qa(qa, self.P["processed"] / "qa_report.json")
        self.log(f"  merged rows: {len(m)}  (dropped: {len(qa.get('dropped', []))})")

    # 03 ------------------------------------------------------------------
    def s03_features(self):
        m = pd.read_csv(self.P["processed"] / "merged_matches.csv", parse_dates=["date"])
        ds, tf, dd = FE.build_dataset(m, self.cfg.STUDY_SEASONS)
        ds.to_csv(self.P["processed"] / "epl_dataset.csv", index=False)
        tf.to_csv(self.P["processed"] / "team_features.csv", index=False)
        dd.to_csv(self.P["processed"] / "data_dictionary.csv", index=False)
        self.log(f"  dataset: {len(ds)} matches x {len(FE.feature_columns())} features")
        self._data = None

    # 04 ------------------------------------------------------------------
    def s04_vif(self):
        ds = pd.read_csv(self.P["processed"] / "epl_dataset.csv", parse_dates=["date"])
        first_test = self.folds[0].test_season
        window = ds[ds["season"] < first_test]
        kept, drops, final = FS.vif_screen(window, FE.feature_columns(), self.cfg.VIF_THRESHOLD)
        json.dump({"features": kept, "fit_seasons": sorted(window["season"].unique().tolist()),
                   "threshold": self.cfg.VIF_THRESHOLD},
                  open(self.P["processed"] / "selected_features.json", "w"), indent=2)
        drops.to_csv(self.P["tables"] / "vif_drop_log.csv", index=False)
        final.rename("vif").to_csv(self.P["tables"] / "vif_final.csv")
        self.log(f"  kept {len(kept)} of {len(FE.feature_columns())} features")
        self._data = None

    # 05-08 ---------------------------------------------------------------
    def s08_benchmark(self, models=None):
        data = self.data()
        for name in models or self.cfg.MODELS:
            EX.run_model(name, data, self.folds, self.cfg, self.P, self.tracker, self.log)
        self._write_results()

    def _write_results(self):
        # every finished model, not just the ones in this call
        bench = EX.load_benchmarks(self.P, config.MODELS)
        if not bench:
            return None
        r = EX.results_tables(bench)
        t = self.P["tables"]
        r["predictions"].to_csv(t / "predictions.csv", index=False)
        r["season"].to_csv(t / "model_metrics_season.csv", index=False)
        r["overall"].to_csv(t / "model_metrics_overall.csv", index=False)
        r["per_class"].to_csv(t / "per_class_metrics.csv", index=False)
        r["confusion"].to_csv(t / "confusion_matrices.csv", index=False)
        r["hardest_by_model"].to_csv(t / "hardest_class_by_model.csv", index=False)
        r["class_f1_mean"].to_csv(t / "class_f1_mean_across_models.csv", index=False)
        rows = []
        for m, b in bench.items():
            for f in b["folds"]:
                rows.append({"model": m, "fold": f["fold"], "test_season": f["test_season"],
                             "epochs": f["epochs"], "val_log_loss": f["val_log_loss"],
                             "params": json.dumps(f["params"])})
        pd.DataFrame(rows).to_csv(t / "best_params.csv", index=False)
        return r

    # 09 ------------------------------------------------------------------
    def s09_stats(self):
        season = pd.read_csv(self.P["tables"] / "model_metrics_season.csv")
        fr, rk, pw = ST.run_tests(season)
        fr.to_csv(self.P["tables"] / "stat_tests_friedman.csv", index=False)
        rk.to_csv(self.P["tables"] / "stat_tests_ranks.csv", index=False)
        pw.to_csv(self.P["tables"] / "stat_tests_pairwise.csv", index=False)
        self.log(fr.to_string(index=False))

    # 10 ------------------------------------------------------------------
    def top_two(self):
        o = pd.read_csv(self.P["tables"] / "model_metrics_overall.csv")
        return o.sort_values("rps_season_mean")["model"].head(2).tolist()

    def s10_shap(self, models=None):
        models = models or self.top_two()
        self.log(f"  explaining: {models}")
        data = self.data()
        res = {m: XP.run_shap(m, data, self.folds, self.P, self.log) for m in models}
        local, glob, cat = XP.shap_tables(res)
        t = self.P["tables"]
        try:
            local.to_parquet(t / "shap_local.parquet", index=False)
        except Exception:
            pass
        local.to_csv(t / "shap_local.csv", index=False)
        glob.to_csv(t / "shap_global.csv", index=False)
        cat.to_csv(t / "shap_category.csv", index=False)
        XP.beeswarm_plots(res, self.P["figures"])

    # 11 ------------------------------------------------------------------
    def s11_ablation(self, models=None):
        data = self.data()
        bench = EX.load_benchmarks(self.P, config.MODELS)
        for name in models or self.cfg.MODELS:
            if name not in bench:
                raise RuntimeError(f"run the benchmark for {name} before its ablation")
            AB.run_ablation(name, data, self.folds, bench[name], self.cfg,
                            self.P, self.tracker, self.log)
        done = [m for m in config.MODELS if (self.P["checkpoints"] / f"ablation_{m}.pkl").exists()]
        abl = pd.concat([pd.read_pickle(self.P["checkpoints"] / f"ablation_{m}.pkl") for m in done],
                        ignore_index=True)
        abl.to_csv(self.P["tables"] / "ablation_by_fold.csv", index=False)
        AB.summarise(abl).to_csv(self.P["tables"] / "ablation_summary.csv", index=False)

    # 12 ------------------------------------------------------------------
    def s12_powerbi(self):
        ds = pd.read_csv(self.P["processed"] / "epl_dataset.csv", parse_dates=["date"])
        dd = pd.read_csv(self.P["processed"] / "data_dictionary.csv")
        kept = json.load(open(self.P["processed"] / "selected_features.json"))["features"]
        dd["retained"] = dd["feature"].isin(kept)
        tables = {}
        p = self.P["tables"] / "predictions.csv"
        if p.exists():
            tables["predictions"] = pd.read_csv(p)
        PB.export(self.P, ds, dd, tables, self.log)



STAGES = ["collect", "merge", "features", "vif", "benchmark", "stats", "shap",
          "ablation", "powerbi"]


def run(stages=None, synthetic=False, models=None, cfg=None, log=print):
    cfg = cfg or settings(MODELS=models)
    pl = Pipeline(cfg, log)
    for st in stages or STAGES:
        log(f"== {st}")
        if st == "collect":
            if not synthetic:
                pl.s01_collect()
        elif st == "merge":
            pl.s02_merge(synthetic)
        elif st == "features":
            pl.s03_features()
        elif st == "vif":
            pl.s04_vif()
        elif st == "benchmark":
            pl.s08_benchmark()
        elif st == "stats":
            pl.s09_stats()
        elif st == "shap":
            pl.s10_shap()
        elif st == "ablation":
            pl.s11_ablation()
        elif st == "powerbi":
            pl.s12_powerbi()
        else:
            raise ValueError(f"unknown stage {st}")
    return pl
