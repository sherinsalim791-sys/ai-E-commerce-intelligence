"""Streamlit dashboard consuming the FastAPI backend."""
import os
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="E-Commerce Intelligence", page_icon="🛒", layout="wide")


def api(path, method="GET", json_body=None, params=None):
    try:
        r = requests.request(method, f"{API_URL}{path}", json=json_body,
                             params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        st.error(f"API error: {e}")
        return None


# Sidebar
st.sidebar.title("🛒 E-Commerce Intelligence")
h = api("/health")
if h:
    st.sidebar.success(f"✅ Backend online")
    st.sidebar.caption(f"Model: **{h['model_name'] or 'N/A'}**")
    st.sidebar.caption(f"Rows: {h['rows_available']:,}")
else:
    st.sidebar.error("❌ Backend offline")
    st.stop()

page = st.sidebar.radio("Navigate", [
    "📊 Overview", "📈 Trends", "🔮 Predict", "🔎 Explore",
])

# ---------------- Overview ----------------
if page == "📊 Overview":
    st.title("📊 Overview")

    k = api("/kpis")
    if k:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("💰 Revenue", f"${k['total_revenue']:,.0f}")
        c2.metric("📈 Profit", f"${k['total_profit']:,.0f}")
        c3.metric("🧾 Orders", f"{k['total_orders']:,}")
        c4.metric("👥 Customers", f"{k['unique_customers']:,}")

        c5, c6, c7 = st.columns(3)
        c5.metric("Avg Order Value", f"${k['avg_order_value']:,.2f}")
        c6.metric("Profit Margin", f"{k['avg_profit_margin_pct']}%")
        c7.metric("Delivered Rate", f"{k['delivered_rate_pct']}%")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        rows = api("/revenue-by/category")
        if rows:
            fig = px.bar(x=[r["revenue"] for r in rows], y=[r["name"] for r in rows],
                         orientation="h", title="Revenue by Category",
                         color=[r["revenue"] for r in rows],
                         color_continuous_scale="Viridis")
            fig.update_layout(showlegend=False, coloraxis_showscale=False, height=420)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        rows = api("/revenue-by/region")
        if rows:
            fig = px.bar(x=[r["revenue"] for r in rows], y=[r["name"] for r in rows],
                         orientation="h", title="Revenue by Region",
                         color=[r["revenue"] for r in rows],
                         color_continuous_scale="Magma")
            fig.update_layout(showlegend=False, coloraxis_showscale=False, height=420)
            st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        rows = api("/revenue-by/channel")
        if rows:
            fig = px.pie(names=[r["name"] for r in rows],
                         values=[r["revenue"] for r in rows],
                         title="Sales Channel Mix",
                         color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)

    with col4:
        rows = api("/order-status")
        if rows:
            fig = px.bar(x=[r["status"] for r in rows], y=[r["count"] for r in rows],
                         title="Order Status",
                         color=[r["status"] for r in rows],
                         color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(showlegend=False, height=420)
            st.plotly_chart(fig, use_container_width=True)

# ---------------- Trends ----------------
elif page == "📈 Trends":
    st.title("📈 Monthly Trends")

    rows = api("/monthly-trend")
    if rows:
        periods = [r["period"] for r in rows]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=periods, y=[r["revenue"] for r in rows],
                                 mode="lines+markers", name="Revenue",
                                 line=dict(color="#2E86AB", width=3)))
        fig.add_trace(go.Scatter(x=periods, y=[r["profit"] for r in rows],
                                 mode="lines+markers", name="Profit",
                                 line=dict(color="#E67E22", width=3), yaxis="y2"))
        fig.update_layout(
            title="Revenue & Profit",
            yaxis=dict(title="Revenue"),
            yaxis2=dict(title="Profit", overlaying="y", side="right"),
            hovermode="x unified", height=460,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("🏆 Top Products")
    n = st.slider("Show top N", 5, 30, 10)
    rows = api("/top-products", params={"n": n})
    if rows:
        fig = px.bar(x=[r["revenue"] for r in rows], y=[r["name"] for r in rows],
                     orientation="h", title=f"Top {n} Products by Revenue",
                     color=[r["revenue"] for r in rows],
                     color_continuous_scale="Cividis")
        fig.update_layout(showlegend=False, coloraxis_showscale=False, height=500)
        st.plotly_chart(fig, use_container_width=True)

# ---------------- Predict ----------------
elif page == "🔮 Predict":
    st.title("🔮 High-Value Order Prediction")
    st.caption("Enter order details to classify as high-value (top 25% of sales).")

    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Customer Age", 18, 100, 35)
            quantity = st.number_input("Quantity", 1, 20, 1)
            unit_price = st.number_input("Unit Price ($)", 0.0, 500.0, 25.0)
        with col2:
            discount = st.number_input("Discount %", 0.0, 50.0, 0.0)
            segment = st.selectbox("Customer Segment", ["Regular", "New", "Premium"])
            region = st.selectbox("Region", ["North", "South", "East", "West", "Central"])
        with col3:
            channel = st.selectbox("Sales Channel", ["Website", "Mobile App", "Marketplace"])
            category = st.selectbox("Category",
                                    ["Electronics", "Fashion", "Home & Kitchen",
                                     "Beauty", "Sports", "Books"])
            payment = st.selectbox("Payment Method",
                                   ["Credit Card", "Debit Card", "Upi",
                                    "Net Banking", "Cash On Delivery"])

        submitted = st.form_submit_button("🚀 Predict", use_container_width=True)

    if submitted:
        payload = {
            "Customer_Age": age,
            "Quantity": quantity,
            "Unit_Price": unit_price,
            "Discount_Percent": discount,
            "Discount_Amount": round(unit_price * quantity * discount / 100, 2),
            "Customer_Segment": segment,
            "Region": region,
            "Sales_Channel": channel,
            "Category": category,
            "Payment_Method": payment,
        }
        result = api("/predict", method="POST", json_body=payload)
        if result:
            c1, c2 = st.columns(2)
            c1.metric("Prediction", result["label"])
            c2.metric("Probability", f"{result['probability']:.2%}")
            if result["is_high_value"]:
                st.success("🎉 High-value order predicted")
            else:
                st.info("Standard order predicted")

# ---------------- Explore ----------------
elif page == "🔎 Explore":
    st.title("🔎 Data Explorer")
    import pandas as pd

    rows = api("/revenue-by/category")
    st.markdown("### Category Summary")
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    rows = api("/top-products", params={"n": 30})
    st.markdown("### Top 30 Products")
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)