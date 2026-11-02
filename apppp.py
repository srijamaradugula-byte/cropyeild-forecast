"""
CROP YIELD FORECAST + FARMER ADVISORY ALERT  (single-file project)
Run:  streamlit run app.py
First run automatically: creates practice data -> cleans it -> trains Gradient Boosting -> opens the app.
To use REAL data: save it as  data/raw_field_data.csv  (same column names), delete the 'models' folder, run again.
"""
import html
import json
import os
import urllib.parse

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ------------------------------------------------------------------ settings
RAW_PATH = "data/raw_field_data.csv"
CLEAN_PATH = "data/clean_data.csv"
MODEL_PATH = "models/yield_model.joblib"
METRICS_PATH = "models/metrics.json"
REPORT_PATH = "models/cleaning_report.json"
TESTPRED_PATH = "models/test_predictions.csv"

TARGET = "Yield_quintal_per_acre"
CATEGORICAL = ["District", "Crop", "Season", "Soil_Type"]
NUMERIC = ["Rainfall_mm", "Avg_Temp_C", "Humidity_pct", "Fertilizer_kg_acre",
           "Pesticide_kg_acre", "Irrigation_pct", "NDVI_mid_season"]
FEATURES = CATEGORICAL + NUMERIC

VALID = {"Rainfall_mm": (0, 3000), "Avg_Temp_C": (5, 50), "Humidity_pct": (0, 100),
         "Fertilizer_kg_acre": (0, 500), "Pesticide_kg_acre": (0, 20),
         "Irrigation_pct": (0, 100), "NDVI_mid_season": (0, 1)}


# ------------------------------------------------------------------ step 1: practice data
def make_sample_data():
    """Creates messy practice data that looks like state agriculture department field data."""
    rng = np.random.default_rng(42)
    N = 3000
    districts = ["Hyderabad", "Warangal", "Karimnagar", "Nizamabad", "Khammam", "Nalgonda", "Adilabad", "Medak"]
    crops = ["Rice", "Cotton", "Maize", "Groundnut", "Chilli"]
    base_yield = {"Rice": 22, "Cotton": 7, "Maize": 25, "Groundnut": 8, "Chilli": 12}
    df = pd.DataFrame({
        "District": rng.choice(districts, N),
        "Crop": rng.choice(crops, N),
        "Season": rng.choice(["Kharif", "Rabi", "Summer"], N),
        "Soil_Type": rng.choice(["Clay", "Loam", "Sandy", "Black"], N),
        "Rainfall_mm": rng.uniform(300, 1500, N).round(1),
        "Avg_Temp_C": rng.uniform(20, 38, N).round(1),
        "Humidity_pct": rng.uniform(40, 90, N).round(1),
        "Fertilizer_kg_acre": rng.uniform(20, 150, N).round(1),
        "Pesticide_kg_acre": rng.uniform(0.2, 3, N).round(2),
        "Irrigation_pct": rng.uniform(0, 100, N).round(1),
        "NDVI_mid_season": rng.uniform(0.3, 0.9, N).round(3),
    })
    soil_bonus = df["Soil_Type"].map({"Clay": 1.05, "Loam": 1.10, "Sandy": 0.88, "Black": 1.08})
    season_bonus = df["Season"].map({"Kharif": 1.0, "Rabi": 0.95, "Summer": 0.85})
    rain_effect = 1 - ((df["Rainfall_mm"] - 900) / 1200) ** 2
    temp_effect = 1 - ((df["Avg_Temp_C"] - 28) / 25) ** 2
    y = (df["Crop"].map(base_yield) * soil_bonus * season_bonus
         * (0.6 + 0.4 * rain_effect) * (0.7 + 0.3 * temp_effect)
         * (0.6 + 0.8 * df["NDVI_mid_season"])
         * (0.85 + 0.002 * df["Fertilizer_kg_acre"] + 0.001 * df["Irrigation_pct"]))
    df[TARGET] = (y * rng.normal(1, 0.05, N)).round(2)

    # make it messy like real field data
    for col in ["Rainfall_mm", "Fertilizer_kg_acre", "NDVI_mid_season", "Humidity_pct", "Soil_Type"]:
        df.loc[rng.choice(N, int(N * 0.04), replace=False), col] = np.nan      # missing values
    idx = rng.choice(N, 150, replace=False)
    df.loc[idx, "Crop"] = df.loc[idx, "Crop"].str.upper() + " "                # "RICE "
    df.loc[rng.choice(N, 20, replace=False), "Avg_Temp_C"] = 99.0              # wrong entries
    df.loc[rng.choice(N, 15, replace=False), TARGET] *= 6                      # outliers
    df = pd.concat([df, df.sample(40, random_state=1)], ignore_index=True)     # duplicates
    df.to_csv(RAW_PATH, index=False)


# ------------------------------------------------------------------ step 2: cleaning
def clean_data():
    raw = pd.read_csv(RAW_PATH)
    df = raw.copy()
    report = {"rows_raw": int(len(raw)),
              "missing_before": {c: int(v) for c, v in raw.isna().sum().items() if v > 0}}

    for c in CATEGORICAL:                                   # 1. fix text ("RICE " -> "Rice")
        df[c] = df[c].str.strip().str.title()

    before = len(df)                                        # 2. remove duplicates
    df = df.drop_duplicates()
    report["duplicates_removed"] = int(before - len(df))

    wrong = 0                                               # 3. impossible values -> missing
    for c, (lo, hi) in VALID.items():
        bad = (df[c] < lo) | (df[c] > hi)
        wrong += int(bad.sum())
        df.loc[bad, c] = float("nan")
    report["wrong_entries_fixed"] = wrong

    df = df.dropna(subset=[TARGET])                         # 4. rows without target

    for c in NUMERIC:                                       # 5. fill missing values
        df[c] = df[c].fillna(df.groupby("Crop")[c].transform("median"))
        df[c] = df[c].fillna(df[c].median())
    for c in CATEGORICAL:
        df[c] = df[c].fillna(df[c].mode()[0])

    q1 = df.groupby("Crop")[TARGET].transform(lambda s: s.quantile(0.25))   # 6. yield outliers (IQR)
    q3 = df.groupby("Crop")[TARGET].transform(lambda s: s.quantile(0.75))
    iqr = q3 - q1
    outlier = (df[TARGET] > q3 + 3 * iqr) | (df[TARGET] < q1 - 3 * iqr) | (df[TARGET] <= 0)
    report["yield_outliers_removed"] = int(outlier.sum())
    df = df[~outlier]

    report["rows_clean"] = int(len(df))
    df.to_csv(CLEAN_PATH, index=False)
    json.dump(report, open(REPORT_PATH, "w"), indent=2)


# ------------------------------------------------------------------ step 3: training
def train_model():
    df = pd.read_csv(CLEAN_PATH)
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def make_prep():
        return ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)],
                                 remainder="passthrough", verbose_feature_names_out=False)

    def scores(model):
        p = model.predict(X_test)
        return {"R2": round(float(r2_score(y_test, p)), 4),
                "MAE": round(float(mean_absolute_error(y_test, p)), 3),
                "RMSE": round(float(np.sqrt(mean_squared_error(y_test, p))), 3)}

    base = Pipeline([("prep", make_prep()), ("model", LinearRegression())]).fit(X_train, y_train)
    results = {"Linear Regression (baseline)": scores(base)}

    search = RandomizedSearchCV(
        Pipeline([("prep", make_prep()), ("model", GradientBoostingRegressor(random_state=42))]),
        param_distributions={"model__n_estimators": [200, 300, 500],
                             "model__learning_rate": [0.03, 0.05, 0.1],
                             "model__max_depth": [2, 3, 4],
                             "model__subsample": [0.7, 0.85, 1.0],
                             "model__min_samples_leaf": [1, 5, 10]},
        n_iter=10, cv=3, scoring="r2", random_state=42, n_jobs=1)
    search.fit(X_train, y_train)
    best = search.best_estimator_
    results["Gradient Boosting (final)"] = scores(best)

    pred_test = best.predict(X_test)
    resid = y_test.values - pred_test
    info = {"best_model": "Gradient Boosting (final)",
            "best_params": {k.replace("model__", ""): (v.item() if hasattr(v, "item") else v)
                            for k, v in search.best_params_.items()},
            "cv_r2": round(float(search.best_score_), 4),
            "residual_q10": float(np.quantile(resid, 0.10)),
            "residual_q90": float(np.quantile(resid, 0.90)),
            "results": results}
    joblib.dump(best, MODEL_PATH)
    json.dump(info, open(METRICS_PATH, "w"), indent=2)
    pd.DataFrame({"Actual": y_test.values, "Predicted": pred_test.round(2)}).to_csv(TESTPRED_PATH, index=False)


def ensure_artifacts():
    """Runs steps 1-3 only if the files are missing."""
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    if not os.path.exists(RAW_PATH):
        make_sample_data()
    if not os.path.exists(CLEAN_PATH) or not os.path.exists(REPORT_PATH):
        clean_data()
    if not (os.path.exists(MODEL_PATH) and os.path.exists(METRICS_PATH) and os.path.exists(TESTPRED_PATH)):
        train_model()


# ------------------------------------------------------------------ farmer advisory alert
TEXT = {
    "English": {
        "title": "CROP YIELD ALERT", "crop": "Crop", "district": "District",
        "expected": "Expected yield", "unit": "quintal", "acre": "acre", "acres": "acres", "total": "Total",
        "status": {"GOOD": "Good yield expected.", "NORMAL": "Normal yield expected.",
                   "LOW": "LOW yield expected - take action."},
        "tips_title": "Advice:",
        "tips": {"irrigation": "Increase irrigation.",
                 "fertilizer": "Apply fertilizer as recommended.",
                 "ndvi": "Crop health is low - inspect the field.",
                 "rain_low": "Rainfall is low - arrange water.",
                 "rain_high": "Heavy rain - keep drainage clear.",
                 "heat": "High temperature - irrigate in morning/evening.",
                 "disease": "High humidity - check for pests and disease.",
                 "none": "No special action needed now."},
        "footer": "Contact your agriculture officer for help.",
    },
}


def build_alert(crop, district, pred, avg, area, f, lang="English"):
    """Returns (status, alert_text)."""
    diff = (pred - avg) / avg * 100
    status = "GOOD" if diff >= 10 else "LOW" if diff <= -10 else "NORMAL"
    tips = []
    if f["Irrigation_pct"] < 40: tips.append("irrigation")
    if f["Fertilizer_kg_acre"] < 50: tips.append("fertilizer")
    if f["NDVI_mid_season"] < 0.5: tips.append("ndvi")
    if f["Rainfall_mm"] < 500: tips.append("rain_low")
    if f["Rainfall_mm"] > 1300: tips.append("rain_high")
    if f["Avg_Temp_C"] > 35: tips.append("heat")
    if f["Humidity_pct"] > 80: tips.append("disease")
    if not tips: tips = ["none"]
    t = TEXT[lang]
    lines = [t["title"],
             f'{t["crop"]}: {crop} | {t["district"]}: {district}',
             f'{t["expected"]}: {pred:.1f} {t["unit"]}/{t["acre"]}',
             f'{t["total"]} ({area:g} {t["acres"]}): {pred * area:.0f} {t["unit"]}',
             t["status"][status], t["tips_title"]]
    lines += [f'- {t["tips"][k]}' for k in tips[:3]]
    lines.append(t["footer"])
    return status, "\n".join(lines)


# ------------------------------------------------------------------ Streamlit app
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_csv(path):
    return pd.read_csv(path)


def main():
    st.set_page_config(page_title="Crop Yield Forecast & Advisory", page_icon="🌾", layout="wide")

    if not os.path.exists(MODEL_PATH):
        with st.spinner("First run: preparing data and training the model (about 1 minute)..."):
            ensure_artifacts()
    else:
        ensure_artifacts()

    model = load_model()
    data = load_csv(CLEAN_PATH)
    metrics = json.load(open(METRICS_PATH))
    clean_report = json.load(open(REPORT_PATH))
    test_pred = load_csv(TESTPRED_PATH)
    crop_avg = data.groupby("Crop")[TARGET].mean()

    st.title("🌾 An Intelligent System to Forecast Per-Acre Yield Ahead of Harvest")
    st.caption("Gradient Boosting model trained on cleaned state agriculture department field data. "
               "The forecast is delivered as a farmer advisory alert.")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["🔮 Forecast & Alert", "📂 Batch Alerts", "🧹 Data Cleaning", "📊 Data Explorer", "🧠 Model Performance"])

    # ---------- TAB 1
    with tab1:
        st.sidebar.header("Field details")
        district = st.sidebar.selectbox("District", sorted(data["District"].unique()))
        crop = st.sidebar.selectbox("Crop", sorted(data["Crop"].unique()))
        season = st.sidebar.selectbox("Season", sorted(data["Season"].unique()))
        soil = st.sidebar.selectbox("Soil type", sorted(data["Soil_Type"].unique()))
        area = st.sidebar.number_input("Field area (acres)", 0.5, 1000.0, 5.0, step=0.5)

        st.sidebar.header("Weather & management")
        rain = st.sidebar.slider("Rainfall (mm)", 300.0, 1500.0, 900.0)
        temp = st.sidebar.slider("Average temperature (°C)", 15.0, 45.0, 28.0)
        hum = st.sidebar.slider("Humidity (%)", 30.0, 100.0, 65.0)
        fert = st.sidebar.slider("Fertilizer (kg/acre)", 0.0, 200.0, 80.0)
        pest = st.sidebar.slider("Pesticide (kg/acre)", 0.0, 5.0, 1.0)
        irr = st.sidebar.slider("Irrigated area (%)", 0.0, 100.0, 60.0)
        ndvi = st.sidebar.slider("Crop greenness - NDVI (mid-season)", 0.1, 1.0, 0.65)

        inputs = {"District": district, "Crop": crop, "Season": season, "Soil_Type": soil,
                  "Rainfall_mm": rain, "Avg_Temp_C": temp, "Humidity_pct": hum,
                  "Fertilizer_kg_acre": fert, "Pesticide_kg_acre": pest,
                  "Irrigation_pct": irr, "NDVI_mid_season": ndvi}
        pred = float(model.predict(pd.DataFrame([inputs])[FEATURES])[0])
        low = max(pred + metrics["residual_q10"], 0)
        high = pred + metrics["residual_q90"]
        avg = float(crop_avg[crop])

        c1, c2, c3 = st.columns(3)
        c1.metric("Forecast yield per acre", f"{pred:.2f} quintal")
        c2.metric("Total expected yield", f"{pred * area:.1f} quintal")
        c3.metric("Likely range per acre", f"{low:.1f} - {high:.1f}")
        st.caption(f"Average {crop} yield in the data: {avg:.1f} quintal/acre. "
                   "The range covers 80% of the model's past errors.")

        status, alert = build_alert(crop, district, pred, avg, area, inputs)
        colors = {"GOOD": "#e6f4ea", "NORMAL": "#fff8e1", "LOW": "#fdecea"}
        st.subheader("📱 Farmer advisory alert")
        st.markdown(
            f"<div style='max-width:380px;padding:16px;border-radius:14px;background:{colors[status]};"
            f"color:#222;border:1px solid #bbb;white-space:pre-wrap;font-size:15px;line-height:1.5'>"
            f"{html.escape(alert)}</div>", unsafe_allow_html=True)
        st.write("")
        b1, b2 = st.columns(2)
        with b1:
            st.link_button("📲 Send on WhatsApp", "https://wa.me/?text=" + urllib.parse.quote(alert))
        with b2:
            st.download_button("⬇️ Download alert (.txt)", alert.encode("utf-8"), "yield_alert.txt")

    # ---------- TAB 2
    with tab2:
        st.write("Upload a CSV of many fields to get a forecast and an alert for each one.")
        st.write("Required columns: " + ", ".join(FEATURES) + "  (optional: Area_acres)")
        st.download_button("Download a sample CSV", data[FEATURES].head(5).to_csv(index=False).encode(),
                           "sample_batch.csv")
        up = st.file_uploader("Upload CSV", type="csv")
        if up:
            df_in = pd.read_csv(up)
            missing = [c for c in FEATURES if c not in df_in.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                df_in["Predicted_Yield_per_acre"] = model.predict(df_in[FEATURES]).round(2)
                alerts, statuses = [], []
                for _, r in df_in.iterrows():
                    a = float(r["Area_acres"]) if "Area_acres" in df_in.columns else 1.0
                    base = float(crop_avg.get(str(r["Crop"]), data[TARGET].mean()))
                    s, t = build_alert(r["Crop"], r["District"], r["Predicted_Yield_per_acre"], base, a, r.to_dict())
                    alerts.append(t.replace("\n", " | "))
                    statuses.append(s)
                df_in["Alert_Level"] = statuses
                df_in["Alert_Message"] = alerts
                st.dataframe(df_in)
                st.download_button("Download results", df_in.to_csv(index=False).encode("utf-8"), "batch_alerts.csv")

    # ---------- TAB 3
    with tab3:
        st.subheader("What was done to the raw data")
        r = clean_report
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Raw rows", r["rows_raw"])
        k2.metric("Clean rows", r["rows_clean"])
        k3.metric("Duplicates removed", r["duplicates_removed"])
        k4.metric("Yield outliers removed", r["yield_outliers_removed"])
        st.write(f"Wrong entries fixed: **{r['wrong_entries_fixed']}**")
        st.write("Missing values found in the raw data:")
        st.dataframe(pd.DataFrame(r["missing_before"], index=["Missing count"]).T)
        st.markdown("""
**Cleaning and transformation steps**
1. Fixed spelling and capital letters in text columns (e.g. `RICE ` → `Rice`).
2. Removed duplicate rows.
3. Treated impossible values (e.g. 99 °C) as missing.
4. Filled missing numbers with the median of that crop, and missing text with the most common value.
5. Removed extreme yield outliers (IQR rule, per crop).
6. Converted categories (District, Crop, Season, Soil) to numbers with one-hot encoding inside the model pipeline.
""")

    # ---------- TAB 4
    with tab4:
        st.subheader("Clean dataset preview")
        st.dataframe(data.head(50))
        st.subheader("Average yield by crop")
        st.bar_chart(crop_avg)
        st.subheader("Average yield by district")
        st.bar_chart(data.groupby("District")[TARGET].mean())
        st.subheader("Rainfall vs yield")
        st.scatter_chart(data, x="Rainfall_mm", y=TARGET, color="Crop")

    # ---------- TAB 5
    with tab5:
        st.write(f"**Final model:** {metrics['best_model']}")
        st.write("Best settings found:", metrics["best_params"])
        st.write(f"Cross-validation R²: **{metrics['cv_r2']}**")
        st.dataframe(pd.DataFrame(metrics["results"]).T)
        st.caption("R² closer to 1 is better. MAE and RMSE are in quintal per acre - lower is better.")
        st.subheader("Actual vs predicted yield (test data)")
        st.scatter_chart(test_pred, x="Actual", y="Predicted")
        est = model.named_steps["model"]
        names = model.named_steps["prep"].get_feature_names_out()
        imp = pd.Series(est.feature_importances_, index=names).sort_values(ascending=False).head(10)
        st.subheader("Top factors affecting yield")
        st.bar_chart(imp)


if __name__ == "__main__":
    main()
