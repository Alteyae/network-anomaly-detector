# Cybersecurity Threat Detector

A Streamlit app that trains a classifier on web/network traffic requests and
flags each one as **benign** or an **attack** (brute-force, SQL injection,
port scanning, credential stuffing, and others). Includes pages to explore
the training data, try predictions on new requests (by hand or via CSV
upload), and inspect model performance.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL it prints (usually `http://localhost:8501`).

## Deploy to Streamlit Community Cloud (free)

1. Push this repo to GitHub — it needs `app.py`, `requirements.txt`, and
   `cybersecurity_threat_data.csv` all together, since the data file
   travels with the app.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with
   GitHub.
3. Click **New app**, pick this repository and branch, and set:
   - **Main file path:** `app.py`
4. Click **Deploy**. The first build takes a minute or two (installing
   dependencies); after that it's live at a `*.streamlit.app` URL.

Every time you push a new commit to the branch it's deployed from, the app
redeploys automatically.

## What's in the app

- **Overview** — dataset summary, attack type breakdown, and a sample of
  the raw data.
- **Try a Prediction** — classify a single request you describe by hand
  (user agent, ports, byte counts), or upload a CSV of multiple records
  and download the results.
- **Model Performance** — accuracy/precision/recall/F1/AUC, confusion
  matrix, ROC curve, feature importance, and a table of sample test-set
  predictions.

The model (a `DecisionTreeClassifier` with `class_weight="balanced"`) is
trained once when the app starts and cached, so it isn't retrained on
every click.

## Where the signal actually comes from

The raw dataset (`cybersecurity_threat_data.csv`) has 10,000 rows and a
~96%/4% benign/attack split. Before building anything, each column was
checked individually for how much it actually relates to the label:

| Column | Verdict |
|---|---|
| `src_ip`, `dst_ip` | Dropped — 100% unique per row, no generalizable pattern |
| `protocol`, `is_internal_traffic` | Dropped — attack rate is nearly identical across every value |
| `url` | Dropped — attack rate is the same whether or not the URL contains suspicious keywords (`admin`, `.env`, `?id=`, etc.) or is missing |
| `bytes_sent`, `bytes_received` | Kept, but weak — raw correlation looked promising (0.14) until log-transforming to remove outlier skew dropped it to ~0.02 |
| `user_agent` | **This is where the real signal is.** Requests from known scanning/exploitation tools (`sqlmap`, `zgrab`, `nikto`, `nmap`, and others) have roughly a **10x higher attack rate** (≈30%) than requests from normal browsers or standard HTTP clients (≈3%). |

That finding is why `app.py` engineers a single boolean feature,
`is_known_attack_tool`, from `user_agent` rather than using the raw text
column or the columns that turned out to carry no signal. The trained
model's feature importances (visible on the **Model Performance** page)
confirm it's the strongest predictor, followed by the port and byte-count
columns.

**`class_weight="balanced"`** is used deliberately: without it, the model
just learns to predict "benign" for everything and still scores ~96%
accuracy while catching zero real attacks — the same accuracy trap that
imbalanced classification always risks. Balancing trades some accuracy
for much higher recall, which is the right tradeoff for a security tool
where a missed attack usually costs more than a false alarm an analyst
has to double-check.

## Files

| File | Purpose |
|---|---|
| `app.py` | The Streamlit app itself, including feature engineering |
| `requirements.txt` | Python dependencies for deployment |
| `cybersecurity_threat_data.csv` | Training data |

## A note on the data

This is synthetic data generated for teaching/demo purposes — the IP
addresses, domains (`example.org`, `internal.bank.local`, etc.), and
traffic patterns are not real. No real infrastructure, company names, or
personal data are represented.
