import csv
import json
import time


class Tracker:
    def __init__(self, root, experiment="epl_match_outcome"):
        self.root = root
        self.csv_path = root / "runs_log.csv"
        try:
            import mlflow
            mlflow.set_tracking_uri(f"sqlite:///{root / 'mlflow.db'}")
            mlflow.set_experiment(experiment)
            self.mlflow = mlflow
        except Exception:
            self.mlflow = None

    def log(self, name, params=None, metrics=None, tags=None):
        params, metrics, tags = params or {}, metrics or {}, tags or {}
        if self.mlflow:
            with self.mlflow.start_run(run_name=name):
                self.mlflow.log_params({k: str(v) for k, v in params.items()})
                self.mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
                self.mlflow.set_tags({k: str(v) for k, v in tags.items()})
        new = not self.csv_path.exists()
        with open(self.csv_path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["time", "run", "params", "metrics", "tags"])
            w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), name, json.dumps(params, default=str),
                        json.dumps(metrics), json.dumps(tags)])
