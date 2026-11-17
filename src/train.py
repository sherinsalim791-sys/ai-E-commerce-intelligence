"""Train models with MLflow tracking; export best model."""
import json
import warnings
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score,
)
from src.preprocess import run_pipeline, build_preprocessor
from src.utils import (
    MLRUNS_DIR, MODELS_DIR, save_model, save_json, get_logger,
)

warnings.filterwarnings("ignore")
log = get_logger(__name__)

EXPERIMENT = "ecommerce-high-value-classifier"


def create_target(df: pd.DataFrame, threshold_pct: float = 0.75) -> pd.Series:
    """Predict whether an order is a 'high-value' order."""
    threshold = df["Sales_Amount"].quantile(threshold_pct)
    return (df["Sales_Amount"] >= threshold).astype(int)


def build_models() -> dict:
    return {
        "logreg": LogisticRegression(max_iter=1000, random_state=42),
        "rf": RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
        ),
        "gb": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42
        ),
    }


def evaluate(model, X_test, y_test) -> dict:
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, prob)) if prob is not None else 0.0,
    }


def train(save_best: bool = True) -> dict:
    mlflow.set_tracking_uri(f"file://{MLRUNS_DIR}")
    mlflow.set_experiment(EXPERIMENT)

    log.info("Building processed dataset...")
    df = run_pipeline(save=True)

    y = create_target(df)
    X = df.drop(columns=["Order_ID", "Customer_ID", "Order_Date",
                         "Product_ID", "Product_Name", "Profit_Margin",
                         "Total_Cost", "Unit_Cost", "Profit",
                         "Sales_Amount", "Stock_Available",
                         "Year", "Month", "Quarter", "Day_of_Week", "Week",
                         "Discount_Amount", "Age_Group"], errors="ignore")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    log.info(f"Train: {X_train.shape}, Test: {X_test.shape}, Pos rate: {y.mean():.2%}")

    results = []
    best = {"f1": -1.0, "model_name": None, "run_id": None, "pipeline": None, "metrics": None}

    preprocessor = build_preprocessor()

    for name, clf in build_models().items():
        with mlflow.start_run(run_name=name) as run:
            mlflow.log_param("model", name)
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("n_features_in", X_train.shape[1])

            pipeline = Pipeline([
                ("prep", preprocessor),
                ("clf", clf),
            ])
            pipeline.fit(X_train, y_train)

            metrics = evaluate(pipeline, X_test, y_test)
            for k, v in metrics.items():
                mlflow.log_metric(k, v)

            mlflow.sklearn.log_model(pipeline, artifact_path="model")

            log.info(f"[{name}] metrics: {metrics}")
            results.append({"model": name, "run_id": run.info.run_id, **metrics})

            if metrics["f1"] > best["f1"]:
                best.update({
                    "f1": metrics["f1"],
                    "model_name": name,
                    "run_id": run.info.run_id,
                    "pipeline": pipeline,
                    "metrics": metrics,
                })

    # Export best model
    if save_best and best["pipeline"] is not None:
        save_model(best["pipeline"], "model.pkl")
        save_json(
            {
                "model_name": best["model_name"],
                "run_id": best["run_id"],
                "metrics": best["metrics"],
                "features": list(X_train.columns),
                "target": "is_high_value (Sales_Amount >= 75th percentile)",
            },
            MODELS_DIR / "metadata.json",
        )
        log.info(f"Exported best model -> models/model.pkl ({best['model_name']})")

    return {"best": {k: v for k, v in best.items() if k != "pipeline"},
            "all_results": results}


if __name__ == "__main__":
    summary = train(save_best=True)
    print(json.dumps(summary, indent=2, default=str))