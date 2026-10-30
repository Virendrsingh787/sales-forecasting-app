import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from xgboost import XGBRegressor

ART = Path(__file__).parent / "artifacts"
MONTHS = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
          7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}

st.set_page_config(page_title="XYZ Drug Stores - Sales Forecast",
                   page_icon="📈", layout="wide")


# ---------------------------------------------------------------- loading
@st.cache_resource
def load_model():
    model = XGBRegressor()
    model.load_model(str(ART / "xgb_model.json"))
    return model


@st.cache_data
def load_data():
    store = pd.read_csv(ART / "store.csv")
    history = pd.read_csv(ART / "history.csv", parse_dates=["Date"])
    meta = json.loads((ART / "meta.json").read_text())
    return store, history, meta


# ------------------------------------------------- feature engineering
# Mirrors Steps 2, 5 and 6 of the notebook exactly.
def build_future(store, store_ids, dates, promo, state_hol, school_hol, meta):
    f = pd.MultiIndex.from_product([store_ids, dates], names=["Store", "Date"]).to_frame(index=False)
    f = f.merge(store, on="Store", how="left")

    # business-logic missing-value fills
    f["CompetitionDistance"] = f["CompetitionDistance"].fillna(store["CompetitionDistance"].max() * 2)
    f["CompetitionOpenKnown"] = f["CompetitionOpenSinceMonth"].notna().astype(int)
    for c in ["CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
              "Promo2SinceWeek", "Promo2SinceYear"]:
        f[c] = f[c].fillna(0)
    f["PromoInterval"] = f["PromoInterval"].fillna("None")

    # date features
    f["Year"] = f["Date"].dt.year
    f["Month"] = f["Date"].dt.month
    f["Week"] = f["Date"].dt.isocalendar().week.astype(int)
    f["DayOfWeek"] = f["Date"].dt.dayofweek
    f["IsWeekend"] = f["DayOfWeek"].isin([5, 6]).astype(int)

    f["CompetitionOpenMonths"] = (
        (f["Year"] - f["CompetitionOpenSinceYear"]) * 12
        + (f["Month"] - f["CompetitionOpenSinceMonth"])
    ).clip(lower=0)

    f["Promo2Weeks"] = 0
    m = f["Promo2"] == 1
    f.loc[m, "Promo2Weeks"] = (
        (f.loc[m, "Year"] - f.loc[m, "Promo2SinceYear"]) * 52
        + (f.loc[m, "Week"] - f.loc[m, "Promo2SinceWeek"])
    ).clip(lower=0)

    month_str = f["Month"].map(MONTHS)
    intervals = f["PromoInterval"].str.split(",")
    f["IsPromoMonth"] = [
        1 if (p2 == 1 and ms in (pi or [])) else 0
        for p2, ms, pi in zip(f["Promo2"], month_str, intervals)
    ]

    # scenario inputs (unknown in advance)
    f["Promo"] = int(promo)
    f["StateHoliday"] = state_hol
    f["SchoolHoliday"] = int(school_hol)

    # categorical encoding with the SAME classes used in training
    for col in ["StoreType", "Assortment", "StateHoliday"]:
        mapping = {c: i for i, c in enumerate(meta["encoder_classes"][col])}
        f[col] = f[col].astype(str).map(mapping)
    return f


@st.cache_data(show_spinner=False)
def forecast(store_ids, horizon, promo, state_hol, school_hol):
    store, _, meta = load_data()
    model = load_model()
    start = pd.Timestamp(meta["last_date"]) + pd.Timedelta(days=1)
    dates = pd.date_range(start, periods=horizon, freq="D")
    f = build_future(store, list(store_ids), dates, promo, state_hol, school_hol, meta)
    f["Predicted_Sales"] = np.clip(np.expm1(model.predict(f[meta["features"]])), 0, None).round(2)
    out = f[["Store", "Date", "Predicted_Sales"]].sort_values(["Store", "Date"]).reset_index(drop=True)
    out.insert(0, "Id", out["Store"].astype(str) + "_" + out["Date"].dt.strftime("%Y-%m-%d"))
    return out


# ------------------------------------------------------------------ UI
store, history, meta = load_data()

st.title("📈 XYZ Drug Stores - Sales Forecasting")
st.caption("XGBoost model trained on log-transformed daily sales, validated on a time-based hold-out "
           "(last 6 weeks). Pick a store and a scenario to get a day-by-day forecast.")

with st.sidebar:
    st.header("Forecast settings")
    store_id = st.selectbox("Store", meta["trained_stores"])
    horizon = st.slider("Days to forecast", 7, 42, 42, step=7)
    st.subheader("Scenario assumptions")
    promo = st.toggle("Promo running every day", value=False)
    state_hol = st.selectbox("State holiday", ["0", "a", "b", "c"],
                             format_func=lambda x: {"0": "None", "a": "Public holiday",
                                                    "b": "Easter", "c": "Christmas"}[x])
    school_hol = st.toggle("School holiday", value=False)
    st.info("Promo and holidays are not known in advance, so they are treated as scenario inputs.")

tab_fc, tab_perf, tab_imp, tab_about = st.tabs(
    ["🔮 Forecast", "📊 Model performance", "🧠 Feature importance", "ℹ️ About"])

# ---- Forecast tab
with tab_fc:
    out = forecast((store_id,), horizon, promo, state_hol, school_hol)
    s = store[store["Store"] == store_id].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total forecast", f"{out['Predicted_Sales'].sum():,.0f}")
    c2.metric("Avg / day", f"{out['Predicted_Sales'].mean():,.0f}")
    peak = out.loc[out["Predicted_Sales"].idxmax()]
    c3.metric("Peak day", peak["Date"].strftime("%a %d %b"), f"{peak['Predicted_Sales']:,.0f}")
    c4.metric("Store type / assortment", f"{s['StoreType']} / {s['Assortment']}")

    hist = history[history["Store"] == store_id].set_index("Date")["Sales"].rename("Actual sales")
    pred = out.set_index("Date")["Predicted_Sales"].rename("Forecast")
    chart = pd.concat([hist, pred], axis=1)
    st.subheader(f"Store {store_id}: last 90 days (actual) + next {horizon} days (forecast)")
    st.line_chart(chart, height=380)

    left, right = st.columns([2, 1])
    with left:
        st.subheader("Forecast table")
        st.dataframe(out[["Id", "Date", "Predicted_Sales"]], width="stretch", hide_index=True)
    with right:
        st.subheader("Average by weekday")
        wd = out.assign(Day=out["Date"].dt.day_name()).groupby("Day")["Predicted_Sales"].mean()
        wd = wd.reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        st.bar_chart(wd)

    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Download this store's forecast (CSV)",
                       out.to_csv(index=False), f"store_{store_id}_forecast.csv", "text/csv")
    if d2.button("Generate batch forecast for ALL stores"):
        batch = forecast(tuple(meta["trained_stores"]), horizon, promo, state_hol, school_hol)
        st.download_button("⬇️ Download batch forecast (CSV)", batch.to_csv(index=False),
                           "all_stores_forecast.csv", "text/csv")
        st.success(f"{len(batch):,} rows ready ({len(meta['trained_stores'])} stores x {horizon} days).")

# ---- Performance tab
with tab_perf:
    st.subheader("Hold-out results (last 6 weeks, never seen in training)")
    perf = pd.DataFrame({"RMSPE (lower is better)": meta["rmspe"],
                         "R² (higher is better)": meta["r2"]}).round(4)
    st.dataframe(perf, width="stretch")
    a, b = st.columns(2)
    a.caption("RMSPE"); a.bar_chart(pd.Series(meta["rmspe"]))
    b.caption("R²"); b.bar_chart(pd.Series(meta["r2"]))
    st.caption(f"Train rows: {meta['n_train_rows']:,} | Test rows: {meta['n_test_rows']:,}")

# ---- Importance tab
with tab_imp:
    st.subheader("What drives the predictions?")
    imp = pd.Series(meta["importance"]).sort_values(ascending=False)
    st.bar_chart(imp)

# ---- About tab
with tab_about:
    st.markdown(f"""
**Pipeline:** merge sales + store tables → business-logic missing-value fills → drop closed days →
date & promo-duration features → `log1p(Sales)` target → time-based split → tuned Random Forest /
XGBoost (early stopping) → RMSPE and R² evaluation.

**Data:** a sample of {len(meta['trained_stores'])} stores (full daily history kept per store).
Data ends on **{meta['last_date']}**, so forecasts start the day after.

**Limitations**
- Forecasts are only offered for stores the model was trained on.
- Promo / holiday flags are scenario assumptions, not predictions.
- The model assumes the store is open every forecast day (closed days were removed from training).
""")
