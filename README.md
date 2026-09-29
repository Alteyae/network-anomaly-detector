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
checked individually for how much it actually relates to the label —
this investigation is laid out in full, with visualizations, in
[`notebooks/14_cybersecurity_threat_detector.ipynb`](https://github.com/Alteyae/ml-cst9/blob/main/notebooks/14_cybersecurity_threat_detector.ipynb)
in the companion course repo:

| Column | Verdict |
|---|---|
| `src_ip`, `dst_ip` | Dropped — 100% unique per row, no generalizable pattern |
| `protocol`, `is_internal_traffic` | Dropped — attack rate is nearly identical across every value |
| `url` | Dropped — attack rate is the same whether or not the URL contains suspicious keywords (`admin`, `.env`, `?id=`, etc.) or is missing |
| `timestamp` | Dropped — attack rate is flat across every hour of the day |
| `bytes_sent`, `bytes_received` | Kept, but weak — raw correlation looked promising (0.14) until log-transforming to remove outlier skew dropped it to ~0.02 |
| `user_agent` | **Real signal.** Requests from known scanning/exploitation tools (`sqlmap`, `zgrab`, `nikto`, `nmap`, and others) have roughly a **6-10x higher attack rate** than requests from normal browsers or standard HTTP clients. |
| `dst_port` | **A second, independent real signal.** Traffic to a destination port that's rarely used elsewhere in the data has roughly a **6x higher attack rate** than traffic to a common port (80, 443, 22, etc.) — and this holds even among requests with a completely ordinary user agent. |

Those two findings are why `app.py` engineers `is_known_attack_tool` and
`is_rare_dst_port`, plus a weaker `src_port_freq` feature, rather than
using the raw columns directly. Both port-based features are fit on the
**training split only** (the same leakage rule as scaling a numeric
feature) — a port count that includes test rows would leak a little
information about the test set into training. The trained model's
feature importances (visible on the **Model Performance** page) confirm
`is_known_attack_tool` and `is_rare_dst_port` are the two strongest
predictors.

**Why two independent features matter more than one:** the single
`is_known_attack_tool` feature only catches attacks that self-identify
via a known tool's user agent string — trivially evaded by an attacker
using a normal browser string. Adding `is_rare_dst_port` catches a
different failure mode (unusual port probing) that doesn't depend on the
user agent at all, which is why adding it raised recall from ~64% to
~79% and AUC from ~0.77 to ~0.84.

**`class_weight="balanced"`** is used deliberately: without it, the model
just learns to predict "benign" for everything and still scores ~96%
accuracy while catching zero real attacks — the same accuracy trap that
imbalanced classification always risks. Balancing trades some accuracy
for much higher recall, which is the right tradeoff for a security tool
where a missed attack usually costs more than a false alarm an analyst
has to double-check.

## Honest limitations

This is a teaching/demo project, not a production-ready detector:

- **Precision is low (~0.18).** For every real attack the model correctly
  flags, it also flags several benign requests. In real use this means a
  lot of false alarms to triage.
- **It still misses roughly 1 in 5 real attacks** (recall ~0.79), and the
  attacks it catches are mostly ones that trip one of the two engineered
  features. An attacker avoiding a known tool's default user agent *and*
  a rare port would likely slip through.
- **Only ~400 attack examples total**, spread across 9 attack types —
  some types have under 15 examples, too few to learn a type-specific
  pattern from.
- The dataset is synthetic. A real deployment would need real traffic
  data, and the specific signals that work here (fixed tool user agent
  strings, port rarity) may not transfer directly to a different
  environment.

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
