import os
import json
import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

cfg = yaml.safe_load(open("params.yaml"))
d, t = cfg["data"], cfg["train"]
target = d["target"]

train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")
X_train, y_train = train.drop(columns=[target]), train[target]
X_test, y_test = test.drop(columns=[target]), test[target]

num_cols = X_train.select_dtypes("number").columns.tolist()
cat_cols = X_train.select_dtypes(exclude="number").columns.tolist()

def make_pipeline(estimator):
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ])
    return Pipeline([("pre", pre), ("model", estimator)])

models = {
    "baseline_median": DummyRegressor(strategy="median"),
    "ridge": Ridge(alpha=t["ridge_alpha"]),
    "random_forest": RandomForestRegressor(n_estimators=t["rf_estimators"],
                                           random_state=d["seed"], n_jobs=-1),
    "lightgbm": LGBMRegressor(n_estimators=t["lgbm_estimators"], learning_rate=t["lgbm_lr"],
                              num_leaves=t["lgbm_leaves"], random_state=d["seed"], verbose=-1),
}

mlflow.set_experiment("house-price")
results, best_name, best_pipe, best_rmse = {}, None, None, float("inf")

for name, est in models.items():
    with mlflow.start_run(run_name=name):
        pipe = make_pipeline(est).fit(X_train, y_train)
        pred = pipe.predict(X_test)
        m = {
            "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
            "mae": float(mean_absolute_error(y_test, pred)),
            "r2": float(r2_score(y_test, pred)),
        }
        mlflow.log_param("model", name)
        mlflow.log_metrics(m)
        mlflow.sklearn.log_model(pipe, name="model", skops_trusted_types=["numpy.dtype"])
        results[name] = m
        print(f"{name:16s} {m}")
        if name != "baseline_median" and m["rmse"] < best_rmse:
            best_name, best_pipe, best_rmse = name, pipe, m["rmse"]

os.makedirs("models", exist_ok=True)
joblib.dump(best_pipe, "models/model.pkl")
json.dump({"best_model": best_name, **results[best_name]}, open("metrics.json", "w"), indent=2)
print(f"\nBest: {best_name} (RMSE {best_rmse:,.0f})")