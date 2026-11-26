
import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

BASE_DIR = Path(__file__).resolve().parents[2]
REGISTRY_PATH = BASE_DIR / "models" / "registry" / "model_registry.json"

st.set_page_config(
    page_title="Model Monitoring | TrashNet",
    layout="wide"
)

st.title("Model Monitoring")
st.write("Monitor API health, model uptime and registered model performance.")

# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def load_registry():
    """Load model information from the model registry."""

    with open(REGISTRY_PATH, "r") as file:
        return json.load(file)


def format_uptime(seconds):
    """Convert uptime seconds into days, hours, minutes and seconds."""

    seconds = int(seconds)

    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)

    return f"{days}d {hours}h {minutes}m {seconds}s"


def get_api_data(endpoint):
    """Retrieve information from the FastAPI backend."""

    try:
        response = requests.get(
            f"{API_URL}/{endpoint}",
            timeout=5
        )

        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException:
        return None


# --------------------------------------------------
# REFRESH
# --------------------------------------------------

if st.button("Refresh Monitoring Data"):
    st.rerun()

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

health_data = get_api_data("health")
uptime_data = get_api_data("uptime")

try:
    registry = load_registry()

    production_version = registry["production_version"]
    models = registry["models"]

    production_model = models[production_version]
    metrics = production_model["metrics"]

except (FileNotFoundError, json.JSONDecodeError, KeyError) as error:
    st.error(f"Could not load model registry: {error}")
    st.stop()

st.divider()

# --------------------------------------------------
# SYSTEM STATUS
# --------------------------------------------------

st.subheader("System Status")

col1, col2, col3 = st.columns(3)

with col1:
    if health_data and health_data.get("status") == "healthy":
        st.success("API: Healthy")
    else:
        st.error("API: Unavailable")

with col2:
    if health_data and health_data.get("model_loaded"):
        st.success("Model: Loaded")
    else:
        st.error("Model: Not Loaded")

with col3:
    if uptime_data:
        st.success("Uptime: Available")
    else:
        st.error("Uptime: Unavailable")

# --------------------------------------------------
# PRODUCTION MODEL
# --------------------------------------------------

st.divider()
st.subheader("Production Model")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Production Version",
        production_version
    )

with col2:
    st.metric(
        "Model Status",
        production_model["status"].capitalize()
    )

with col3:
    st.metric(
        "Model File",
        production_model["file"]
    )

if uptime_data:
    st.metric(
        "API Uptime",
        format_uptime(uptime_data["uptime_seconds"])
    )

# --------------------------------------------------
# PRODUCTION PERFORMANCE
# --------------------------------------------------

st.divider()
st.subheader("Production Model Performance")

col1, col2, col3, col4 = st.columns(4)

performance_metrics = [
    ("Accuracy", "accuracy"),
    ("Precision", "precision"),
    ("Recall", "recall"),
    ("Macro-F1", "macro_f1")
]

for column, (label, key) in zip(
    [col1, col2, col3, col4],
    performance_metrics
):
    with column:
        st.metric(
            label,
            f"{metrics[key] * 100:.2f}%"
        )

# --------------------------------------------------
# MODEL REGISTRY
# --------------------------------------------------

st.divider()
st.subheader("Model Registry")

registry_rows = []

for version, model in models.items():

    row = {
    "Version": version,
    "Status": model["status"].capitalize(),
    "Model File": model["file"],
    "Accuracy": model["metrics"]["accuracy"] * 100,
    "Precision": model["metrics"]["precision"] * 100,
    "Recall": model["metrics"]["recall"] * 100,
    "Macro-F1": model["metrics"]["macro_f1"] * 100
    }

    if model.get("reason"):
        row["Reason"] = model["reason"]

    registry_rows.append(row)

registry_df = pd.DataFrame(registry_rows)

st.dataframe(
    registry_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Accuracy": st.column_config.NumberColumn(
            format="%.2f%%",
            help="Accuracy stored as a decimal"
        ),
        "Precision": st.column_config.NumberColumn(
            format="%.2f%%"
        ),
        "Recall": st.column_config.NumberColumn(
            format="%.2f%%"
        ),
        "Macro-F1": st.column_config.NumberColumn(
            format="%.2f%%"
        )
    }
)

# --------------------------------------------------
# MODEL COMPARISON
# --------------------------------------------------

st.divider()
st.subheader("Model Performance Comparison")

comparison_df = pd.DataFrame([
    {
        "Version": version,
        "Accuracy": model["metrics"]["accuracy"] * 100,
        "Macro-F1": model["metrics"]["macro_f1"] * 100,
        "Status": model["status"].capitalize()
    }
    for version, model in models.items()
])

fig = px.bar(
    comparison_df,
    x="Version",
    y=["Accuracy", "Macro-F1"],
    barmode="group",
    title="Performance Across Registered Models",
    labels={
        "value": "Score (%)",
        "variable": "Metric"
    },
    color_discrete_sequence=["#2E86DE", "#20BF6B"]
)

fig.update_layout(
    yaxis_range=[0, 100]
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# --------------------------------------------------
# REJECTED MODEL INFORMATION
# --------------------------------------------------

st.divider()
st.subheader("Rejected Candidates")

rejected_models = {
    version: model
    for version, model in models.items()
    if model["status"] == "rejected"
}

if rejected_models:

    for version, model in rejected_models.items():

        with st.expander(f"Model {version}"):

            st.write(f"**File:** {model['file']}")

            st.write(
                f"**Macro-F1:** "
                f"{model['metrics']['macro_f1'] * 100:.2f}%"
            )

            st.write(
                f"**Accuracy:** "
                f"{model['metrics']['accuracy'] * 100:.2f}%"
            )

            if model.get("reason"):
                st.warning(model["reason"])

else:
    st.info("No rejected candidates registered.")

st.divider()

st.caption(
    "TrashNet MLOps | Model Monitoring Dashboard"
)
