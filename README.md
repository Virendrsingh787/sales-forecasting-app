# 📈 XYZ Drug Stores - Sales Forecasting (Streamlit)

Interactive sales forecasting app. An XGBoost model (trained on `log1p(Sales)`, time-based validation)
predicts daily sales for the next 6 weeks for any store, under user-selected promo / holiday scenarios.

**Live demo:** <PASTE YOUR STREAMLIT URL HERE>

## Highlights
- Business-logic missing-value handling (no blind mean/median fill)
- Date, competition-duration and Promo2 features; log-transformed target
- Time-based train/test split (last 6 weeks held out) - no leakage
- Models: baseline vs Random Forest vs XGBoost (early stopping), scored with RMSPE and R²
- Per-store forecast, scenario simulation, and batch CSV export

## Results
| Model | RMSPE | R² |
|---|---|---|
| Baseline | _fill in_ | _fill in_ |
| Random Forest | _fill in_ | _fill in_ |
| XGBoost | _fill in_ | _fill in_ |

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tech stack
Python, pandas, NumPy, scikit-learn, XGBoost, Streamlit
