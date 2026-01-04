import os
import time
import random
from pathlib import Path

import numpy as np
import optuna
import mlflow
import mlflow.xgboost

from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
)

from src.data_loader import load_and_split_data
from src.objective import objective
from src.evaluate import evaluate_best_model


# ---------------------------
# Environment detection
# ---------------------------
def running_in_docker():
    return os.path.exists("/.dockerenv")


# ---------------------------
# Global configuration
# ---------------------------
SEED = 42
N_TRIALS = 100
N_JOBS = 2
STUDY_NAME = "xgboost-housing-optimization"
MLFLOW_EXPERIMENT = "optuna-xaboost-optimization"

OUTPUT_DIR = Path("/app/outputs") if running_in_docker() else Path("outputs")
STORAGE_URL = f"sqlite:///{OUTPUT_DIR / 'optuna_study.db'}"


def main():
    start_time = time.time()

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Reproducibility
    random.seed(SEED)
    np.random.seed(SEED)

    # ---- MLflow setup (CORRECT on Windows & Docker) ----
    mlruns_path = (OUTPUT_DIR / "mlruns").resolve()
    mlflow.set_tracking_uri(mlruns_path.as_uri())
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    # Load data
    X_train, X_test, y_train, y_test = load_and_split_data(
        test_size=0.2,
        random_state=SEED,
    )

    # Optuna study
    sampler = TPESampler(seed=SEED)
    pruner = MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=5,
    )

    study = optuna.create_study(
        study_name=STUDY_NAME,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=STORAGE_URL,
        load_if_exists=True,
    )

    def objective_with_logging(trial):
        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
            try:
                score = objective(trial, X_train, y_train)

                mlflow.log_params(trial.params)

                cv_mse = -score
                cv_rmse = np.sqrt(cv_mse)

                mlflow.log_metric("cv_mse", cv_mse)
                mlflow.log_metric("cv_rmse", cv_rmse)
                mlflow.log_metric("trial_number", trial.number)

                mlflow.set_tag("trial_state", "COMPLETE")
                return score

            except optuna.TrialPruned:
                mlflow.set_tag("trial_state", "PRUNED")
                raise

            except Exception:
                mlflow.set_tag("trial_state", "FAIL")
                raise

    study.optimize(
        objective_with_logging,
        n_trials=N_TRIALS,
        n_jobs=N_JOBS,
        show_progress_bar=False,
    )

    # Save Optuna visualizations
    plot_optimization_history(study).write_image(
        OUTPUT_DIR / "optimization_history.png"
    )
    plot_param_importances(study).write_image(
        OUTPUT_DIR / "param_importance.png"
    )

    # Final evaluation
    evaluate_best_model(
        study=study,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        output_dir=str(OUTPUT_DIR),
        start_time=start_time,
    )


if __name__ == "__main__":
    main()
