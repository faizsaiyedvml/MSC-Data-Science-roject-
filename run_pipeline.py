import argparse

from epl import pipeline


def main():
    ap = argparse.ArgumentParser(description="EPL match outcome pipeline")
    ap.add_argument("--stages", nargs="+", choices=pipeline.STAGES)
    ap.add_argument("--models", nargs="+", choices=["logreg", "rf", "xgb", "gb", "lstm"])
    ap.add_argument("--synthetic", action="store_true", help="use the synthetic league")
    ap.add_argument("--fast", action="store_true", help="3 trials, 20 LSTM epochs (testing only)")
    ap.add_argument("--n-iter", type=int)
    ap.add_argument("--out", help="output folder")
    ap.add_argument("--retune-ablation", action="store_true")
    ap.add_argument("--likert", help="Likert CSV (runs stage 13 only)")
    a = ap.parse_args()

    over = {"MODELS": a.models, "OUT_DIR": a.out, "N_ITER": a.n_iter}
    if a.fast:
        over.update(N_ITER=a.n_iter or 3, LSTM_MAX_EPOCHS=20, LSTM_PATIENCE=3)
    if a.retune_ablation:
        over["ABLATION_RETUNE"] = True
    cfg = pipeline.settings(**over)

    if a.likert:
        pipeline.Pipeline(cfg).s13_likert(a.likert)
        return
    pipeline.run(a.stages, synthetic=a.synthetic, cfg=cfg)


if __name__ == "__main__":
    main()
