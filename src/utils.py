"""Shared helpers — paths, logging, model IO."""
from pathlib import Path
import json
import logging
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_RAW = BASE_DIR / "data" / "raw" / "ecommerceintelligence.csv"
DATA_PROCESSED = BASE_DIR / "data" / "processed" / "cleaned_data.csv"
MODELS_DIR = BASE_DIR / "models"
MLRUNS_DIR = BASE_DIR / "mlruns"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.parent.mkdir(parents=True, exist_ok=True)

DATE_COL = "Order_Date"
TARGET_COL = "Sales_Amount"
PROFIT_COL = "Profit"

NUMERIC_COLS = [
    "Customer_Age", "Quantity", "Unit_Price", "Discount_Percent",
    "Sales_Amount", "Unit_Cost", "Total_Cost", "Profit", "Stock_Available",
]
CATEGORICAL_COLS = [
    "Customer_Segment", "Region", "City", "Sales_Channel",
    "Category", "Payment_Method", "Order_Status",
]


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger(name)


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def save_model(model, name: str = "model.pkl") -> Path:
    path = MODELS_DIR / name
    joblib.dump(model, path)
    return path


def load_model(name: str = "model.pkl"):
    path = MODELS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)