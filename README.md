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

The model (a `RandomForestClassifier` with a tuned `class_weight`) is
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
user agent at all.

## Choosing the model

With the features settled, several models and `class_weight` settings
were compared on the same held-out test set (see the notebook, Section
5a, for the full comparison table and chart):

| Model | Precision | Recall | F1 | AUC |
|---|---|---|---|---|
| `DecisionTreeClassifier(class_weight="balanced")` | 0.18 | 0.79 | 0.30 | 0.84 |
| `RandomForestClassifier(class_weight={0:1,1:3})` | 0.39 | 0.25 | 0.31 | 0.86 |
| **`RandomForestClassifier(class_weight={0:1,1:5})`** | **0.32** | **0.58** | **0.41** | **0.86** |
| `RandomForestClassifier(class_weight={0:1,1:8})` | 0.27 | 0.71 | 0.39 | 0.86 |

The `{0:1,1:5}` RandomForest is what's deployed: it wins on F1 (the best
balance of precision and recall) and roughly **doubles precision** versus
the original `"balanced"` DecisionTree, while still catching more than
half of real attacks. `class_weight="balanced"` weights the minority
class by its exact inverse frequency (about 24:1 here) — a milder,
manually chosen ratio turned out to work better than that automatic
setting for this specific dataset.

**Two ideas that were tested and explicitly ruled out** (also in the
notebook, Section 5a) — worth knowing so they aren't retried from
scratch:
- A stricter rarity threshold on `dst_port` doesn't help — almost every
  "rare" port in this dataset already appears only once, for benign and
  attack rows alike, so there's no frequency gradient left to exploit.
- A "touches multiple rare ports" scanning-behavior feature isn't
  buildable on this dataset, since every `src_ip` appears in exactly one
  row (no repeated-visitor structure to detect a burst from).

## Honest limitations

This is a teaching/demo project, not a production-ready detector:

- **Precision is still modest (~0.32).** For every two real attacks the
  model correctly flags, it also flags roughly four benign requests. In
  real use this means false alarms to triage, even though it's about
  twice as precise as the first version of this model.
- **It misses close to half of real attacks** (recall ~0.58) — this
  specific model was chosen to trade some of the earlier version's recall
  (~0.79) for much better precision. A model tuned the other way
  (`class_weight={0:1,1:8}`, in the comparison table above) is available
  if catching more attacks matters more than fewer false alarms for a
  given use case.
- The attacks it catches are mostly ones that trip one of the two
  engineered features. An attacker avoiding a known tool's default user
  agent *and* a rare port would likely slip through.
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
