import os
import json
import time
import subprocess
from datetime import datetime
from typing import Dict, Any, Optional

class ExperimentTracker:
    """
    Lightweight, reproducible Experiment Tracker for logging hyperparameters,
    git commit hashes, dataset statistics, and validation/test metrics.
    """
    def __init__(self, experiment_name: str, base_dir: str = "experiments/runs"):
        self.experiment_name = experiment_name
        self.base_dir = base_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{experiment_name}_{self.timestamp}"
        self.run_dir = os.path.join(self.base_dir, self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)
        self.git_commit = self._get_git_commit()
        self.metadata: Dict[str, Any] = {
            "experiment_name": experiment_name,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "git_commit": self.git_commit,
            "hyperparameters": {},
            "metrics": {},
            "notes": ""
        }

    def _get_git_commit(self) -> str:
        try:
            res = subprocess.run(["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            return res.stdout.strip()
        except Exception:
            return "unknown_commit"

    def log_hyperparameters(self, params: Dict[str, Any]) -> None:
        self.metadata["hyperparameters"].update(params)
        self.save()

    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        self.metadata["metrics"].update(metrics)
        self.save()

    def log_notes(self, notes: str) -> None:
        self.metadata["notes"] = notes
        self.save()

    def save(self) -> str:
        json_path = os.path.join(self.run_dir, "run_summary.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        return json_path
