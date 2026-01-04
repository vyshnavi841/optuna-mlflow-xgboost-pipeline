import numpy as np

from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor


def objective(trial, X_train, y_train):
    """
    Optuna objective function for XGBoost regression.
    Returns negative mean squared error.
    """

    # EXACTLY 7 hyperparameters (as required)
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 0.5),
    }

    model = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=1,
        **params,
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="neg_mean_squared_error",
        n_jobs=1,
    )

    return np.mean(scores)
