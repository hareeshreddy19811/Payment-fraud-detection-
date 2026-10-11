"""
Online Payment Fraud Detection (Streamlit app)
-----------------------------------------------
Trains a Random Forest classifier on the PaySim dataset (step, type, amount)
and lets the user check whether a transaction looks fraudulent.

Run:
    streamlit run app.py

Dataset:
    Place the full PaySim CSV at data/PS_20174392719_1491204439457_log.csv
    (https://www.kaggle.com/datasets/ealaxi/paysim1). If it is missing, the app
    falls back to the small synthetic data/sample_transactions.csv so it still runs.
"""

import os

import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="Online Payment Fraud Detection", page_icon="🔐")

st.title("🔐 Online Payment Fraud Detection")
st.write("Detect potentially fraudulent online payment transactions.")

DATA_DIR = "data"
FULL_DATASET = os.path.join(DATA_DIR, "PS_20174392719_1491204439457_log.csv")
SAMPLE_DATASET = os.path.join(DATA_DIR, "sample_transactions.csv")
MAX_ROWS = 200_000  # limit sample size for faster training


def resolve_dataset():
    """Return (path, is_full_dataset), or (None, False) if nothing is available."""
    if os.path.exists(FULL_DATASET):
        return FULL_DATASET, True
    if os.path.exists(SAMPLE_DATASET):
        return SAMPLE_DATASET, False
    return None, False


@st.cache_resource
def train_model(file_path: str):
    data = pd.read_csv(file_path)
    data = data[["step", "type", "amount", "isFraud"]].dropna()

    if len(data) > MAX_ROWS:
        data = data.sample(n=MAX_ROWS, random_state=42)

    X = data[["step", "type", "amount"]]
    y = data["isFraud"].astype(int)

    if y.nunique() < 2 or y.value_counts().min() < 2:
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocess = ColumnTransformer(
        [
            ("num", StandardScaler(), ["step", "amount"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["type"]),
        ]
    )

    model = Pipeline(
        [
            ("preprocess", preprocess),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    report = classification_report(
        y_test, predictions, output_dict=True, zero_division=0
    )
    return model, report


path, is_full = resolve_dataset()

if path is None:
    st.error(
        "No dataset found. Download the PaySim CSV and place it in the "
        "`data/` folder."
    )
    st.markdown(
        "[Download dataset from Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1)"
    )
    st.stop()

result = train_model(path)

if result is None:
    st.error("The dataset does not contain enough fraud examples to train a model.")
    st.stop()

model, report = result

st.success("Machine learning model trained successfully!")
if not is_full:
    st.warning(
        "Using the small synthetic sample dataset. Add the full PaySim CSV to "
        "`data/` for realistic results."
    )

st.subheader("Enter Transaction Details")

transaction_type = st.selectbox(
    "Payment Type", ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"]
)
amount = st.number_input(
    "Transaction Amount", min_value=0.0, value=1000.0, step=100.0
)
step = st.number_input(
    "Transaction Time Step (hour)", min_value=1, max_value=744, value=100
)

if st.button("Detect Fraud"):
    transaction = pd.DataFrame(
        [{"step": step, "type": transaction_type, "amount": amount}]
    )

    prediction = model.predict(transaction)[0]
    probabilities = model.predict_proba(transaction)[0]
    fraud_index = list(model.classes_).index(1)
    fraud_score = probabilities[fraud_index]

    st.metric("Estimated Fraud Score", f"{fraud_score:.2%}")

    if prediction == 1:
        st.error("Potential Fraud Detected!")
        st.write("This transaction has been flagged for review.")
    else:
        st.success("Transaction Classified as Likely Legitimate")

    st.caption(
        "This is a machine-learning estimate, not a real bank "
        "verification. Do not use it alone to approve or reject payments."
    )

with st.expander("View Model Evaluation"):
    st.write("Evaluation on the held-out test dataset")
    st.write(pd.DataFrame(report).transpose().round(3))
