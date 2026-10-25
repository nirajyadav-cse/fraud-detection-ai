from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "fraud_model.joblib"
FEATURES_PATH = ROOT / "model_features.json"

st.set_page_config(page_title="AI Fraud Detection", page_icon="🛡️", layout="wide")
st.title("🛡️ AI Fraud Detection in Financial Transactions")
st.caption("Educational demo only. Predictions are estimates, not proof of fraud.")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
        return None, None
    return joblib.load(MODEL_PATH), json.loads(FEATURES_PATH.read_text())

model, features = load_model()
with st.sidebar:
    st.header("About")
    st.write("Upload a CSV of transaction-like records to estimate which rows may be suspicious.")
    st.warning("This demo uses synthetic data. Do not upload real customer or payment data.")

if model is None:
    st.error("Model files are missing. Run `python train_model.py`, then refresh this page.")
    st.stop()

st.subheader("Upload transactions")
st.write("Required columns: " + ", ".join(f"`{x}`" for x in features))
sample_path = ROOT / "sample_transactions.csv"
if sample_path.exists():
    st.download_button("Download sample CSV", sample_path.read_bytes(), "sample_transactions.csv", "text/csv")

uploaded = st.file_uploader("Choose a transaction CSV", type=["csv"])
if uploaded:
    try:
        data = pd.read_csv(uploaded)
        st.write("Preview", data.head())
        missing = [c for c in features if c not in data.columns]
        if missing:
            st.error("Missing required columns: " + ", ".join(missing))
        else:
            X = data[features].apply(pd.to_numeric, errors="coerce")
            if X.isna().any().any():
                st.error("Required feature columns contain empty or non-numeric values.")
            elif len(data) == 0:
                st.warning("The CSV contains no transaction rows.")
            else:
                probs = model.predict_proba(X)[:, 1]
                labels = (probs >= 0.5).astype(int)
                result = data.copy()
                result["fraud_probability"] = probs.round(4)
                result["predicted_label"] = [
                    "Review - potentially fraudulent" if label else "Likely legitimate"
                    for label in labels
                ]
                a, b, c = st.columns(3)
                a.metric("Transactions analyzed", len(result))
                b.metric("Flagged for review", int(labels.sum()))
                c.metric("Flag rate", f"{labels.mean() * 100:.1f}%")
                st.subheader("Prediction results")
                st.dataframe(result, use_container_width=True)
                st.download_button("Download predictions CSV", result.to_csv(index=False).encode(), "fraud_predictions.csv", "text/csv")
                st.info("Flagged transactions require human review. False positives and false negatives are possible.")
    except Exception as exc:
        st.error(f"Could not read the CSV: {exc}")
else:
    st.info("Upload a CSV or download the sample CSV to try the dashboard.")
    st.markdown("**Workflow:** synthetic training data → Random Forest model → CSV upload → risk estimates.")
