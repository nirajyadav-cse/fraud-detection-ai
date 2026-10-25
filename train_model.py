from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score

ROOT = Path(__file__).resolve().parent
FEATURES = ["amount", "time", "merchant_risk", "distance_from_home_km", "failed_attempts_24h", "is_international"]
TARGET = "is_fraud"

def make_demo_data(n=12000, seed=42):
    """Generate synthetic examples for education; these are not real transactions."""
    rng = np.random.default_rng(seed)
    amount = np.clip(rng.lognormal(3.2, 1.0, n), 1, 10000)
    time = rng.integers(0, 24, n)
    merchant_risk = rng.beta(2, 5, n)
    distance = np.clip(rng.gamma(2.0, 30.0, n), 0, 2000)
    failed = np.clip(rng.poisson(0.25, n), 0, 8)
    international = rng.binomial(1, 0.08, n)
    logit = (-5.0 + 0.00025 * amount + 2.3 * merchant_risk
             + 0.0015 * distance + 0.65 * failed + 1.25 * international
             + 0.65 * ((time <= 4) | (time >= 23)).astype(float))
    probability = 1 / (1 + np.exp(-logit))
    fraud = rng.binomial(1, probability)
    return pd.DataFrame({
        "amount": amount.round(2), "time": time,
        "merchant_risk": merchant_risk.round(4),
        "distance_from_home_km": distance.round(2),
        "failed_attempts_24h": failed, "is_international": international,
        TARGET: fraud
    })

def main():
    df = make_demo_data()
    df.to_csv(ROOT / "synthetic_training_data.csv", index=False)
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=250, class_weight="balanced", random_state=42,
            min_samples_leaf=3, n_jobs=-1
        ))
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    print("Evaluation on synthetic hold-out data:")
    print(classification_report(y_test, pred, zero_division=0))
    print(f"ROC-AUC: {roc_auc_score(y_test, probs):.3f}")
    joblib.dump(model, ROOT / "fraud_model.joblib")
    (ROOT / "model_features.json").write_text(json.dumps(FEATURES, indent=2))
    make_demo_data(30, seed=2026).drop(columns=[TARGET]).to_csv(ROOT / "sample_transactions.csv", index=False)
    print("Created model files and sample_transactions.csv")

if __name__ == "__main__":
    main()
