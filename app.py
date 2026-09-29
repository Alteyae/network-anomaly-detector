"""
Network Traffic Anomaly Detector
=================================
Trains a classifier on network/web traffic records (label: 0 = benign, 1 = attack),
then lets a user explore the model's performance and try it on new traffic records --
either typed in by hand or uploaded as a CSV.

Feature engineering note: the raw dataset's strongest signal turned out to live in the
user_agent string (known attack/scanning tools have a far higher attack rate than normal
browsers) rather than in the raw numeric columns, which were checked and found to carry
almost no signal on their own (see README for the numbers). is_known_attack_tool below is
built from that finding.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, roc_auc_score,
)

RANDOM_STATE = 42
# Resolve relative to this file, not the process's working directory -- Streamlit
# Cloud runs the app with the repo root as cwd, which isn't guaranteed to match
# wherever this script happens to be invoked from.
DATA_PATH = Path(__file__).parent / "cybersecurity_threat_data.csv"
TARGET_COLUMN = "label"

# Case-insensitive substrings that flag a request as coming from a known scanning /
# exploitation tool rather than a normal browser or standard HTTP client.
KNOWN_ATTACK_TOOLS = [
    "sqlmap", "zgrab", "nikto", "nmap", "masscan", "hydra",
    "gobuster", "dirbuster", "wpscan",
]

FEATURE_COLUMNS = ["is_known_attack_tool", "src_port", "dst_port", "bytes_sent", "bytes_received"]

st.set_page_config(page_title="Cybersecurity Threat Detector", page_icon="🛡️", layout="wide")


# ----------------------------------------------------------------------------
# Feature engineering
# ----------------------------------------------------------------------------

def engineer_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Add is_known_attack_tool from user_agent; keep the numeric columns as-is.

    src_ip / dst_ip are dropped (100% unique per row -- pure noise, not a
    generalizable pattern), and url / protocol / is_internal_traffic are dropped
    because they showed no meaningful relationship with the label when checked.
    """
    df = raw_df.copy()
    df["is_known_attack_tool"] = (
        df["user_agent"].fillna("").str.lower()
        .apply(lambda ua: any(tool in ua for tool in KNOWN_ATTACK_TOOLS))
        .astype(int)
    )
    return df


# ----------------------------------------------------------------------------
# Data + model (cached so the app doesn't retrain on every click)
# ----------------------------------------------------------------------------

@st.cache_data
def load_data() -> pd.DataFrame:
    raw_df = pd.read_csv(DATA_PATH)
    return engineer_features(raw_df)


@st.cache_resource
def train_pipeline(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # class_weight="balanced" matters a lot here: with only ~4% of rows being
    # attacks, a model trained without it just learns to predict "benign" every
    # time and still scores high accuracy. Balancing trades some accuracy for
    # actually catching attacks (higher recall) -- the right tradeoff for a
    # security tool, where a missed attack is usually worse than a false alarm.
    model = DecisionTreeClassifier(max_depth=4, class_weight="balanced", random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc": roc_auc_score(y_test, y_proba),
    }

    return {
        "model": model,
        "model_name": "Decision Tree (class-balanced)",
        "feature_columns": FEATURE_COLUMNS,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "metrics": metrics,
        "class_balance": y.value_counts(normalize=True).to_dict(),
    }


df = load_data()
pipeline = train_pipeline(df)


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------

st.sidebar.title("Cybersecurity Threat Detector")
st.sidebar.markdown(
    "Classifies web/network traffic requests as **benign** or **attack** using a "
    "model trained on request metadata and the client's user agent."
)
st.sidebar.markdown(f"**Model in use:** {pipeline['model_name']}")
st.sidebar.markdown(f"**Training rows:** {len(df):,}")
st.sidebar.markdown(
    f"**Class balance:** {pipeline['class_balance'].get(0, 0):.0%} benign / "
    f"{pipeline['class_balance'].get(1, 0):.0%} attack"
)
st.sidebar.caption("A scikit-learn classifier pipeline, deployed with Streamlit.")

page = st.sidebar.radio("Go to", ["Overview", "Try a Prediction", "Model Performance"])


# ----------------------------------------------------------------------------
# Page: Overview
# ----------------------------------------------------------------------------

if page == "Overview":
    st.title("🛡️ Cybersecurity Threat Detector")
    st.markdown(
        "This app trains a classifier on web/network request records to flag each "
        "one as **benign** or an **attack** (e.g. brute-force, SQL injection, "
        "port scanning, credential stuffing)."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{len(df):,}")
    col2.metric("Attack rate", f"{pipeline['class_balance'].get(1, 0):.1%}")
    col3.metric("Attack types", f"{df['attack_type'].nunique() - 1}")

    st.subheader("Attack type breakdown")
    st.bar_chart(df[df["attack_type"] != "benign"]["attack_type"].value_counts())

    st.subheader("Sample of the training data")
    st.dataframe(df.head(20), width="stretch")

    st.subheader("Where the model's signal actually comes from")
    st.info(
        "The raw numeric columns (ports, byte counts) and the URL path barely relate "
        "to the label on their own. The one feature that does carry real signal is "
        "**`user_agent`**: requests from known scanning/exploitation tools (e.g. "
        "`sqlmap`, `zgrab`, `nikto`) have a roughly **10x higher attack rate** than "
        "requests from normal browsers or standard HTTP clients. That single "
        "engineered feature (`is_known_attack_tool`) is doing most of the work here "
        "-- see **Model Performance** for the honest numbers, including where the "
        "model still gets it wrong."
    )

# ----------------------------------------------------------------------------
# Page: Try a Prediction
# ----------------------------------------------------------------------------

elif page == "Try a Prediction":
    st.title("Try a Prediction")
    st.markdown("Describe a traffic request, or upload a CSV of records, to classify.")

    tab1, tab2 = st.tabs(["Manual entry", "Upload CSV"])

    with tab1:
        st.markdown("Fill in the request details below.")
        col_a, col_b = st.columns(2)
        with col_a:
            user_agent = st.text_input(
                "User agent string",
                value="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                help="Try entering 'sqlmap/1.8' or 'zgrab/0.x' to see how a known attack tool is flagged.",
            )
            src_port = st.number_input("Source port", min_value=0, max_value=65535, value=51000)
            dst_port = st.number_input("Destination port", min_value=0, max_value=65535, value=443)
        with col_b:
            bytes_sent = st.number_input("Bytes sent", min_value=0, value=20000)
            bytes_received = st.number_input("Bytes received", min_value=0, value=35000)

        if st.button("Classify this request", type="primary"):
            is_tool = int(any(tool in user_agent.lower() for tool in KNOWN_ATTACK_TOOLS))
            record = pd.DataFrame([{
                "is_known_attack_tool": is_tool,
                "src_port": src_port,
                "dst_port": dst_port,
                "bytes_sent": bytes_sent,
                "bytes_received": bytes_received,
            }])[pipeline["feature_columns"]]

            pred = pipeline["model"].predict(record)[0]
            proba = pipeline["model"].predict_proba(record)[0, 1]

            label = "🚨 Attack" if pred == 1 else "✅ Benign"
            st.subheader(f"Prediction: {label}")
            st.metric("Predicted attack probability", f"{proba:.1%}")
            st.progress(min(max(proba, 0.0), 1.0))
            if is_tool:
                st.caption("Flagged: user agent matches a known scanning/exploitation tool.")

    with tab2:
        st.markdown(
            "Upload a CSV with these columns: `user_agent`, `src_port`, `dst_port`, "
            "`bytes_sent`, `bytes_received`."
        )
        uploaded = st.file_uploader("Upload traffic records (CSV)", type="csv")
        if uploaded is not None:
            new_df = pd.read_csv(uploaded)
            required_raw_cols = {"user_agent", "src_port", "dst_port", "bytes_sent", "bytes_received"}
            missing_cols = required_raw_cols - set(new_df.columns)
            if missing_cols:
                st.error(f"Missing required columns: {sorted(missing_cols)}")
            else:
                new_df["is_known_attack_tool"] = (
                    new_df["user_agent"].fillna("").str.lower()
                    .apply(lambda ua: any(tool in ua for tool in KNOWN_ATTACK_TOOLS))
                    .astype(int)
                )
                X_new = new_df[pipeline["feature_columns"]]
                new_df["predicted_label"] = pipeline["model"].predict(X_new)
                new_df["attack_probability"] = pipeline["model"].predict_proba(X_new)[:, 1]
                new_df["prediction"] = new_df["predicted_label"].map({0: "benign", 1: "attack"})

                n_attacks = int((new_df["predicted_label"] == 1).sum())
                st.success(f"Classified {len(new_df)} rows -- {n_attacks} flagged as attacks.")
                st.dataframe(
                    new_df.drop(columns=["predicted_label"]).style.map(
                        lambda v: "background-color: #ffcccc" if v == "attack" else "",
                        subset=["prediction"],
                    ),
                    width="stretch",
                )
                st.download_button(
                    "Download results as CSV",
                    new_df.to_csv(index=False).encode("utf-8"),
                    "predictions.csv",
                    "text/csv",
                )

# ----------------------------------------------------------------------------
# Page: Model Performance
# ----------------------------------------------------------------------------

elif page == "Model Performance":
    st.title("Model Performance")
    st.markdown(
        f"Evaluated on a held-out test set (never seen during training) using the "
        f"**{pipeline['model_name']}** model."
    )

    m = pipeline["metrics"]
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Accuracy", f"{m['accuracy']:.3f}")
    col2.metric("Precision", f"{m['precision']:.3f}")
    col3.metric("Recall", f"{m['recall']:.3f}")
    col4.metric("F1 Score", f"{m['f1']:.3f}")
    col5.metric("AUC", f"{m['auc']:.3f}")

    st.info(
        "This model is tuned to catch attacks (high recall) at the cost of some false "
        "alarms (lower precision) -- deliberately, since `class_weight='balanced'` was "
        "used because a missed attack is usually more costly than an analyst double-"
        "checking a benign request. An AUC well above 0.5 confirms the model is "
        "genuinely separating the two classes, not just guessing."
    )

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(pipeline["y_test"], pipeline["y_pred"])
        fig_cm, ax_cm = plt.subplots(figsize=(4.5, 4))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["benign", "attack"])
        disp.plot(cmap="Blues", ax=ax_cm, colorbar=False)
        st.pyplot(fig_cm)

    with col_b:
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(pipeline["y_test"], pipeline["y_proba"])
        fig_roc, ax_roc = plt.subplots(figsize=(4.5, 4))
        ax_roc.plot(fpr, tpr, label=f"model (AUC = {m['auc']:.3f})")
        ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray", label="random guessing")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.legend()
        st.pyplot(fig_roc)

    st.subheader("Feature importance")
    importances = pd.Series(
        pipeline["model"].feature_importances_, index=pipeline["feature_columns"]
    ).sort_values(ascending=False)
    fig_imp, ax_imp = plt.subplots(figsize=(6, 3))
    importances.plot(kind="barh", ax=ax_imp, color="#4c72b0")
    ax_imp.invert_yaxis()
    ax_imp.set_xlabel("Importance")
    st.pyplot(fig_imp)

    st.subheader("Sample test-set predictions")
    sample_idx = pipeline["X_test"].index[:10]
    sample_display = pipeline["X_test"].loc[sample_idx].copy()
    sample_display["true_label"] = pipeline["y_test"].loc[sample_idx].map({0: "benign", 1: "attack"})
    sample_display["predicted_label"] = (
        pd.Series(pipeline["y_pred"], index=pipeline["X_test"].index)
        .loc[sample_idx].map({0: "benign", 1: "attack"})
    )
    st.dataframe(sample_display, width="stretch")
