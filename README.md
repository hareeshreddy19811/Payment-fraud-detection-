# Online Payment Fraud Detection

A Streamlit web app that uses a Random Forest classifier to flag potentially
fraudulent online payment transactions based on payment type, amount, and
time step.

## What it does

- Loads the PaySim transaction dataset (`step`, `type`, `amount`, `isFraud`)
- Trains a scikit-learn pipeline (scaling + one-hot encoding + `RandomForestClassifier`)
- Handles class imbalance with `class_weight="balanced"` and a stratified split
- Lets you enter a transaction and returns a fraud score and a flag/legitimate verdict
- Shows precision, recall and F1 on a held-out test set

## Project structure

```
payment-fraud-detection/
├── app.py                        # Streamlit app
├── data/
│   └── sample_transactions.csv   # Small synthetic dataset so the app runs out of the box
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
git clone https://github.com/<your-username>/payment-fraud-detection.git
cd payment-fraud-detection
pip install -r requirements.txt
```

## Dataset

For realistic results, download the PaySim dataset from
[Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1) and place it at:

```
data/PS_20174392719_1491204439457_log.csv
```

The file is about 470 MB, so it is git-ignored and must not be committed
(GitHub rejects files over 100 MB). If it is missing, the app falls back to
`data/sample_transactions.csv`, a small synthetic dataset, and shows a warning.
Training uses at most 200,000 sampled rows for speed.

## Usage

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (usually http://localhost:8501),
choose a payment type, amount and time step, and click **Detect Fraud**.

## Disclaimer

This is a machine-learning demo, not a real bank verification system. Do not
use it alone to approve or reject payments.

## License

MIT
