<<<<<<< HEAD
# E-Commerce Intelligence — MLflow + FastAPI + Streamlit

End-to-end ML system with:
- **Preprocessing** → `src/preprocess.py`
- **Training with MLflow tracking** → `src/train.py`
- **FastAPI backend** → `backend/main.py`
- **Streamlit frontend** → `frontend/app.py`
- **Docker compose** deployment

## Quick Start (local)

```bash
# 1. Install
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt

# 2. Preprocess + Train (creates models/model.pkl and mlruns/)
python -m src.train

# 3. Inspect MLflow UI
mlflow ui --backend-store-uri ./mlruns
# Open http://localhost:5000

# 4. Start backend
uvicorn backend.main:app --reload --port 8000
# Docs: http://localhost:8000/docs

# 5. Start frontend (new terminal)
streamlit run frontend/app.py
# Open http://localhost:8501
=======
# ai-E-commerce-intelligence
AI Powered E-Commerce Intelligence system demand + customer+product analytics
>>>>>>> a90737be12eee65cde855b16ea7246e0f896e6ab
