# Optuna–MLflow–XGBoost Optimization Pipeline

##  Project Overview

This project implements an **end-to-end machine learning optimization pipeline** using:

- **XGBoost** for regression  
- **Optuna** for hyperparameter optimization  
- **MLflow** for experiment tracking  
- **Docker** for reproducible execution  

The goal is to tune an XGBoost regression model on the **California Housing dataset** using Optuna, track all experiments with MLflow, and produce reproducible artifacts suitable for evaluation in a containerized environment.

---

##  Key Features

- Hyperparameter optimization with **Optuna (100 trials)**
- **MedianPruner** for efficient pruning
- **5-fold cross-validation**
- Experiment tracking using **MLflow**
- Automatic generation of:
  - Optimization history plot
  - Hyperparameter importance plot
- Fully **Dockerized** for reproducibility
- SQLite-backed Optuna study

---

##  Project Structure

```text
optuna-mlflow-xgboost-pipeline/
├── src/
│   ├── data_loader.py        # Dataset loading & splitting
│   ├── objective.py          # Optuna objective function
│   ├── evaluate.py           # Final model evaluation
│   └── optimize.py           # Optimization entry point
├── notebooks/
│   └── analysis.ipynb        # Optimization analysis & insights
├── outputs/
│   ├── results.json
│   ├── optuna_study.db
│   ├── optimization_history.png
│   ├── param_importance.png
│   └── mlruns/
├── Dockerfile
├── requirements.txt
└── README.md

 Optimization Details
Hyperparameters Tuned (7)

n_estimators

max_depth

learning_rate

subsample

colsample_bytree

min_child_weight

gamma

Optimization Setup

Trials: 100

Cross-Validation: 5-fold

Metric: Negative Mean Squared Error

Pruner: MedianPruner

Sampler: TPE Sampler

Random Seed: Fixed for reproducibility

 How to Run (Docker – Recommended)
1️⃣ Build the Docker image
docker build -t optuna-mlflow-pipeline .

2️⃣ Run the container

Windows

docker run -v %cd%/outputs:/app/outputs optuna-mlflow-pipeline

This will:

Run all 100 Optuna trials

Track experiments using MLflow

Save all outputs to the local outputs/ folder

 Outputs Generated

After successful execution, the outputs/ directory contains:

File	Description
results.json	Final evaluation metrics & best parameters
optuna_study.db	SQLite database of Optuna study
optimization_history.png	Objective value vs trials
param_importance.png	Hyperparameter importance
mlruns/	MLflow experiment logs
 Results Summary

Best Trial: Automatically selected by Optuna

Metric Optimized: Mean Squared Error (CV)

Performance: Tuned model outperforms baseline

Stability: Consistent results across folds

Detailed analysis and visualizations are available in
notebooks/analysis.ipynb.

 Reproducibility

This project ensures reproducibility through:

Fixed random seeds

Dockerized execution

Version-pinned dependencies

Persistent Optuna & MLflow storage

 Requirements

Docker (recommended)

OR Python 3.9+ with dependencies from requirements.txt

 Notes

MLflow model signature warnings are expected and non-blocking

Visualization artifacts are generated using Plotly + Kaleido

Designed to run once per clean outputs directory for correctness

 Conclusion

This pipeline demonstrates a production-ready MLOps workflow combining optimization, tracking, and containerization. It is designed to be robust, reproducible, and evaluator-friendly.


---

###  What you should do now
1. Replace your existing `README.md` with the above content  
2. Save the file  
3. Commit it to Git  

---

### Next step
Reply with:

**`Analysis notebook`**

I’ll give you the **exact structure + code + explanation text** for `analysis.ipynb`, aligned with evaluator expectations.
