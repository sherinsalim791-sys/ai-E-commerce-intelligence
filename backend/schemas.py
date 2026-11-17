"""Pydantic schemas."""
from typing import Optional, List
from pydantic import BaseModel, Field


class OrderFeatures(BaseModel):
    Customer_Age: int = Field(..., ge=18, le=100)
    Quantity: int = Field(..., ge=1)
    Unit_Price: float = Field(..., ge=0)
    Discount_Percent: float = Field(..., ge=0, le=100)
    Discount_Amount: float = Field(..., ge=0)
    Customer_Segment: str
    Region: str
    Sales_Channel: str
    Category: str
    Payment_Method: str


class PredictionResponse(BaseModel):
    is_high_value: int
    probability: float
    label: str
    model_version: str


class KPIResponse(BaseModel):
    total_orders: int
    unique_customers: int
    total_revenue: float
    total_profit: float
    avg_order_value: float
    avg_profit_margin_pct: float
    delivered_rate_pct: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: Optional[str] = None
    rows_available: int