# 📈 Stores: Sales Forecasting

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-app-ff4b4b)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-green)
![Status](https://img.shields.io/badge/status-active-brightgreen)

An interactive web app that forecasts **daily sales for any of 1,115 stores** for the next 6 weeks.
Pick a store, set a promotion / holiday scenario, and get a day-by-day forecast with charts and CSV export.

**🔗 Live demo:** <PASTE YOUR STREAMLIT URL HERE>

<!-- Add a screenshot: save it as docs/screenshot.png and uncomment the line below -->
<!-- ![App screenshot](docs/screenshot.png) -->

---

## Table of contents
- [Business problem](#business-problem)
- [Features](#features)
- [Approach](#approach)
- [Results](#results)
- [Project structure](#project-structure)
- [Getting started](#getting-started)
- [How to use the app](#how-to-use-the-app)
- [Retraining the model](#retraining-the-model)
- [Limitations](#limitations)
- [Tech stack](#tech-stack)
- [Author](#author)

---

## Business problem
Store managers need to know how much each store will sell in the coming weeks to:

- **Plan staffing**: schedule more people on busy days
- **Manage inventory**: avoid empty shelves and overstock
- **Evaluate promotions**: see how a promotion could change sales
- **Set realistic targets**: build budgets and sales goals

This project turns historical sales data into easy-to-read forecasts that non-technical users can explore.

## Features
- 🔮 **Per-store forecast** for 7 to 42 days ahead
- 🎛️ **Scenario simulation**: toggle promotions, state holidays and school holidays
- 📊 **Interactive charts**: recent actual sales plus forecast, and average sales by weekday
- 📥 **CSV export** for a single store or a **batch forecast for all 1,115 stores**
- 📉 **Model performance** and **feature importance** tabs
- ⚠️ **Reliability warning** for stores the model did not train on

## Approach
1. **Data merge**: combine daily sales records with the store details table.
2. **Missing-value handling with business logic**: no blind mean/median fills. For example, a missing competitor distance means "no nearby competitor", so it is filled with a large value, and missing Promo2 dates mean "not participating".
3. **Cleaning**: drop days when the store was closed (zero sales are not demand).
4. **Feature engineering**: date features (year, month, week, day of week, weekend), months since competition opened, weeks since Promo2 started, and whether the current month is a Promo2 month.
5. **Target transform**: train on `log1p(Sales)` to handle skewed sales, then convert back with `expm1`.
6. **Time-based split**: the **last 6 weeks are held out** for testing, so there is no future-data leakage.
7. **Modelling**: baseline vs Random Forest vs XGBoost (with early stopping).
8. **Evaluation**: RMSPE (the metric used in the original Rossmann competition) and R².

## Results

| Model | RMSPE ↓ | R² ↑ |
|---|---|---|
| Baseline | _fill in_ | _fill in_ |
| Random Forest | _fill in_ | _fill in_ |
| XGBoost (final) | _fill in_ | _fill in_ |

> Scores are measured on the last 6 weeks of data, which the model never saw during training.
> The same numbers are shown live in the app's **Model performance** tab.

## Project structure
```
sales-forecast/
├── app.py                # Streamlit app (UI, feature building, forecasting)
├── export_artifacts.py   # Notebook cell that saves the model and data the app needs
├── artifacts/
│   ├── xgb_model.json    # trained XGBoost model
│   ├── store.csv         # store details (all 1,115 stores)
│   ├── history.csv       # last 90 days of actual sales (training sample stores)
│   └── meta.json         # features, encoders, scores, feature importances
├── requirements.txt
└── README.md
```

## Getting started

### Prerequisites
- Python 3.10 or newer
- Git

### Installation
```bash
# 1. Clone the repository
git clone https://github.com/Virendrsingh787/sales-forecasting-app.git
cd sales-forecasting-app

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```
The app opens at **http://localhost:8501**.

## How to use the app
1. **Select a store** in the left panel.
2. **Choose how many days** to forecast (7 to 42).
3. **Set the scenario**: promotion running every day, state holiday, school holiday.
4. **Read the results**: total forecast, average per day, peak day, chart, and daily table.
5. **Download** the forecast as CSV, or generate a batch forecast for all stores.

## Retraining the model
1. Open the training notebook and remove or change the store sampling step to train on more stores.
2. Run all cells, then run the code in `export_artifacts.py` as the last notebook cell.
3. Unzip the downloaded `artifacts.zip` and replace the `artifacts/` folder in this repo.
4. Make sure the `xgboost` version in `requirements.txt` matches the version printed by the export cell.
5. Commit and push:
   ```bash
   git add .
   git commit -m "Retrain model"
   git push
   ```

## Limitations
- **Training sample**: the model was trained on a sample of stores. Forecasts for other stores are available but are **less reliable** (the app shows a warning).
- **Scenario inputs**: promotions and holidays are your assumptions, not predictions.
- **Open days only**: the model assumes the store is open every forecast day, since closed days were removed from training.
- **Forecast horizon**: accuracy is best for the 6-week window the model was validated on.
- **External factors**: weather, local events and sudden market changes are not modelled.

## Tech stack
Python · pandas · NumPy · scikit-learn · XGBoost · Streamlit

## Author
**Virendra Singh**
GitHub: [@Virendrsingh787](https://github.com/Virendrsingh787)

---
⭐ If you found this project useful, consider giving it a star.