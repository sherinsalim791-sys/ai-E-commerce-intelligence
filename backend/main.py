"""FastAPI backend for E-Commerce Intelligence."""
from pathlib import Path
from typing import Optional, List
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import load_model, load_json, MODELS_DIR, DATA_PROCESSED, DATA_RAW
from backend.schemas import (
    OrderFeatures, PredictionResponse, KPIResponse, HealthResponse,
)

app = FastAPI(title="E-Commerce Intelligence API", version="1.0.0")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ---------- Startup: load model + data ----------
MODEL = None
METADATA = {}

@app.on_event("startup")
def startup():
    global MODEL, METADATA
    try:
        MODEL = load_model("model.pkl")
        METADATA = load_json(MODELS_DIR / "metadata.json")
        print(f"[startup] Model loaded: {METADATA.get('model_name')}")
    except FileNotFoundError:
        print("[startup] No trained model found — run src/train.py first")


def load_df() -> pd.DataFrame:
    path = DATA_PROCESSED if DATA_PROCESSED.exists() else DATA_RAW
    return pd.read_csv(path, parse_dates=["Order_Date"])


# ---------- Routes ----------
@app.get("/", tags=["root"])
def root():
    return {"service": "ecommerce-intelligence", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse, tags=["health"])
def health():
    try:
        rows = len(load_df())
    except Exception:
        rows = 0
    return HealthResponse(
        status="ok",
        model_loaded=MODEL is not None,
        model_name=METADATA.get("model_name"),
        rows_available=rows,
    )


@app.get("/kpis", response_model=KPIResponse, tags=["analytics"])
def kpis():
    df = load_df()
    delivered = df[df["Order_Status"].str.title() == "Delivered"]
    return KPIResponse(
        total_orders=int(df["Order_ID"].nunique()),
        unique_customers=int(df["Customer_ID"].nunique()),
        total_revenue=round(float(df["Sales_Amount"].sum()), 2),
        total_profit=round(float(df["Profit"].sum()), 2),
        avg_order_value=round(float(df["Sales_Amount"].mean()), 2),
        avg_profit_margin_pct=round(float((delivered["Profit"] / delivered["Sales_Amount"]).mean() * 100), 2),
        delivered_rate_pct=round(len(delivered) / len(df) * 100, 2),
    )


@app.get("/revenue-by/{dimension}", tags=["analytics"])
def revenue_by(dimension: str):
    allowed = {
        "category": "Category",
        "region": "Region",
        "channel": "Sales_Channel",
        "segment": "Customer_Segment",
        "payment": "Payment_Method",
    }
    if dimension not in allowed:
        raise HTTPException(400, f"dimension must be one of {list(allowed)}")
    df = load_df()
    g = (df.groupby(allowed[dimension])
           .agg(revenue=("Sales_Amount", "sum"),
                profit=("Profit", "sum"),
                orders=("Order_ID", "nunique"))
           .reset_index()
           .rename(columns={allowed[dimension]: "name"})
           .sort_values("revenue", ascending=False))
    return g.round(2).to_dict(orient="records")


@app.get("/monthly-trend", tags=["analytics"])
def monthly_trend():
    df = load_df()
    s = df.set_index("Order_Date").resample("MS").agg(
        revenue=("Sales_Amount", "sum"),
        profit=("Profit", "sum"),
        orders=("Order_ID", "nunique"),
    ).reset_index()
    s["period"] = s["Order_Date"].dt.strftime("%Y-%m")
    return s[["period", "revenue", "profit", "orders"]].round(2).to_dict(orient="records")


@app.get("/top-products", tags=["analytics"])
def top_products(n: int = Query(10, ge=1, le=50)):
    df = load_df()
    g = (df.groupby("Product_Name")
           .agg(revenue=("Sales_Amount", "sum"),
                units=("Quantity", "sum"))
           .nlargest(n, "revenue")
           .reset_index()
           .rename(columns={"Product_Name": "name"}))
    return g.round(2).to_dict(orient="records")


@app.get("/order-status", tags=["analytics"])
def order_status():
    df = load_df()
    s = df["Order_Status"].str.title().value_counts().reset_index()
    s.columns = ["status", "count"]
    return s.to_dict(orient="records")


@app.post("/predict", response_model=PredictionResponse, tags=["ml"])
def predict(features: OrderFeatures):
    if MODEL is None:
        raise HTTPException(503, "Model not loaded — run src/train.py first")

    row = pd.DataFrame([features.model_dump()])
    prob = float(MODEL.predict_proba(row)[0, 1])
    pred = int(prob >= 0.5)
    return PredictionResponse(
        is_high_value=pred,
        probability=round(prob, 4),
        label="High-Value" if pred else "Standard",
        model_version=METADATA.get("model_name", "unknown"),
    )