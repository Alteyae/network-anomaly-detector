# Network Traffic Anomaly Detector

A Streamlit app that trains a classifier on network traffic data and
classifies records as **normal** or **anomalous**. Includes pages to
explore the training data, try predictions on new records (by hand or via
CSV upload), and inspect model performance.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL it prints (usually `http://localhost:8501`).

## Deploy to Streamlit Community Cloud (free)

1. Push this repo to GitHub — it needs `app.py`, `requirements.txt`, and
   `network_traffic_data.csv` all together, since the data file travels
   with the app.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with
   GitHub.
3. Click **New app**, pick this repository and branch, and set:
   - **Main file path:** `app.py`
4. Click **Deploy**. The first build takes a minute or two (installing
   dependencies); after that it's live at a `*.streamlit.app` URL.

Every time you push a new commit to the branch it's deployed from, the app
redeploys automatically.

## What's in the app

- **Overview** — dataset summary and a sample of the raw data.
- **Try a Prediction** — classify a single record you fill in by hand, or
  upload a CSV of multiple records and download the results.
- **Model Performance** — accuracy/precision/recall/F1/AUC, confusion
  matrix, ROC curve, and a table of sample test-set predictions.

The model (a `DecisionTreeClassifier` or `LogisticRegression`, whichever
scores better after `GridSearchCV` tuning) is trained once when the app
starts and cached, so it isn't retrained on every click.

## Files

| File | Purpose |
|---|---|
| `app.py` | The Streamlit app itself |
| `requirements.txt` | Python dependencies for deployment |
| `network_traffic_data.csv` | Training data |

## A note on this specific dataset

This particular copy of the dataset has very little recoverable signal
between its features and the `label` column, so the trained model mostly
predicts the majority class ("normal"). The app surfaces this honestly on
the **Model Performance** page rather than hiding it. To get a genuinely
useful anomaly detector, swap in a dataset where the features actually
relate to the target — the app's structure (load → train → cache →
predict) stays the same; only `network_traffic_data.csv` and
`TARGET_COLUMN` in `app.py` need to change.
