"""Cleaning & feature engineering — reused by train.py and backend."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.utils import (
    DATA_RAW, DATA_PROCESSED, DATE_COL,
    NUMERIC_COLS, CATEGORICAL_COLS, get_logger,
)

log = get_logger(__name__)


def load_raw(path=DATA_RAW) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[DATE_COL])
    log.info(f"Loaded raw data: {df.shape}")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    before = len(df)
    df.drop_duplicates(subset=["Order_ID"], inplace=True)
    log.info(f"Dropped {before - len(df)} duplicate orders")

    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["Customer_Age"] = df["Customer_Age"].fillna(df["Customer_Age"].median())
    for c in CATEGORICAL_COLS:
        df[c] = df[c].astype(str).str.strip().str.title().fillna("Unknown")

    df = df[(df["Quantity"] > 0) & (df["Unit_Price"] > 0) & (df["Sales_Amount"] >= 0)]
    df = df[(df["Discount_Percent"] >= 0) & (df["Discount_Percent"] <= 100)]

    return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Year"] = df[DATE_COL].dt.year
    df["Month"] = df[DATE_COL].dt.month
    df["Quarter"] = df[DATE_COL].dt.quarter
    df["Day_of_Week"] = df[DATE_COL].dt.dayofweek
    df["Week"] = df[DATE_COL].dt.isocalendar().week.astype(int)

    df["Profit_Margin"] = np.where(
        df["Sales_Amount"] > 0, df["Profit"] / df["Sales_Amount"], 0
    )
    df["Discount_Amount"] = (
        df["Unit_Price"] * df["Quantity"] * df["Discount_Percent"] / 100
    )
    df["Age_Group"] = pd.cut(
        df["Customer_Age"],
        bins=[0, 25, 35, 45, 55, 100],
        labels=["18-25", "26-35", "36-45", "46-55", "55+"],
    ).astype(str)
    return df


def run_pipeline(save: bool = True) -> pd.DataFrame:
    df = load_raw()
    df = clean(df)
    df = engineer_features(df)
    if save:
        df.to_csv(DATA_PROCESSED, index=False)
        log.info(f"Saved processed data -> {DATA_PROCESSED}")
    return df


def build_preprocessor() -> ColumnTransformer:
    """ColumnTransformer for classification models (churn / high-value)."""
    num_features = ["Customer_Age", "Quantity", "Unit_Price",
                "Discount_Percent"]
    cat_features = ["Customer_Segment", "Region", "Sales_Channel",
                    "Category", "Payment_Method"]
    return ColumnTransformer([
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
    ])