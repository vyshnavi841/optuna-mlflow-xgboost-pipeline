import json
import time
import numpy as np
import mlflow
import mlflow.xgboost

from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from optuna.trial import TrialState


def evaluate_best_model(
    study,
    X_train,
    X_test,
    y_train,
    y_test,
    output_dir,
    start_time,
):
    """
    Train and evaluate the best XGBoost model found by Optuna.
    Logs results to MLflow and writes structured results.json.

    Parameters
    ----------
    study : optuna.study.Study
        Completed Optuna study
    X_train, X_test : np.ndarray
        Training and test features
    y_train, y_test : np.ndarray
        Training and test targets
    output_dir : str
        Directory where outputs are written
    start_time : float
        Optimization start timestamp
    """

    best_params = study.best_params

    # Train best model on full training data
    best_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=1,
        **best_params,
    )

    best_model.fit(X_train, y_train)

    # Test set evaluation
    y_pred = best_model.predict(X_test)

    test_mse = mean_squared_error(y_test, y_pred)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test, y_pred)

    # Final MLflow run for best model
    with mlflow.start_run(run_name="best_model"):
        mlflow.set_tag("best_model", "true")

        # Log hyperparameters
        mlflow.log_params(best_params)

        # Log test metrics
        mlflow.log_metric("test_mse", test_mse)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("test_r2", test_r2)

        # Save trained model
        mlflow.xgboost.log_model(
            best_model,
            artifact_path="model"
        )

    # Study statistics
    n_trials_completed = len(
        [t for t in study.trials if t.state == TrialState.COMPLETE]
    )
    n_trials_pruned = len(
        [t for t in study.trials if t.state == TrialState.PRUNED]
    )

    optimization_time_seconds = time.time() - start_time

    # Structured results.json
    results = {
        "n_trials_completed": n_trials_completed,
        "n_trials_pruned": n_trials_pruned,
        "best_cv_rmse": float(np.sqrt(-study.best_value)),
        "test_rmse": float(test_rmse),
        "test_r2": float(test_r2),
        "best_params": best_params,
        "optimization_time_seconds": float(optimization_time_seconds),
    }

    results_path = f"{output_dir}/results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
