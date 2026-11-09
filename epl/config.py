import os
from pathlib import Path

SEED = 42
N_ITER = 20                 # sampled configurations per model per fold
MIN_TRAIN_SEASONS = 3

# season = starting year, so 2015 means 2015/16
WARMUP_SEASON = 2014        # used to seed cold-start values
STUDY_SEASONS = list(range(2015, 2026))   # 2015/16 .. 2025/26
# Elo is calculated from results (api.clubelo.com returned 502 when tested, Sept 2026).
ELO_BURNIN_SEASONS = 5      # 2009/10 .. 2013/14
ELO_K = 20.0
ELO_HOME_ADV = 60.0
ELO_START = 1500.0

SEQ_LEN = 5
VIF_THRESHOLD = 10.0
DATE_TOLERANCE_DAYS = 3

MODELS = ["logreg", "rf", "xgb", "gb", "lstm"]

LSTM_MAX_EPOCHS = 100
LSTM_PATIENCE = 10

# ablation reuses each fold's tuned full-model parameters unless this is True
ABLATION_RETUNE = False

if Path("/kaggle/working").exists():
    _default_out = "/kaggle/working/epl_output"
else:
    _default_out = str(Path.cwd() / "epl_output")
OUT_DIR = Path(os.environ.get("EPL_OUT_DIR", _default_out))


def paths(out_dir=None):
    root = Path(out_dir) if out_dir else OUT_DIR
    p = {
        "root": root,
        "raw": root / "data" / "raw",
        "processed": root / "data" / "processed",
        "checkpoints": root / "checkpoints",
        "models": root / "checkpoints" / "models",
        "tables": root / "tables",
        "figures": root / "figures",
        "powerbi": root / "powerbi_exports",
    }
    for v in p.values():
        v.mkdir(parents=True, exist_ok=True)
    return p


def in_cloud():
    return (
        bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))
        or Path("/kaggle").exists()
        or bool(os.environ.get("COLAB_RELEASE_TAG"))
    )
