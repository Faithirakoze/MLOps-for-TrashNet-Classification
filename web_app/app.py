
import os
import streamlit as st
import requests

# Page configuration
st.set_page_config(
    page_title="TrashNet MLOps",
    layout="wide"
)

# FastAPI backend URL
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# Application title
st.title("TrashNet MLOps Dashboard")

st.write(
    "An intelligent waste classification system "
    "with automated model monitoring and retraining."
)

st.divider()

# Backend status
st.subheader("System Status")

try:
    response = requests.get(f"{API_URL}/health", timeout=5)

    if response.status_code == 200:
        st.success("FastAPI backend is running.")
    else:
        st.error("Backend is responding with an error.")

except requests.exceptions.RequestException:
    st.error("Backend is unavailable. Start the FastAPI server.")

# Navigation information
st.subheader("Application Features")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Waste Prediction")
    st.write("Upload an image and classify it into one of six waste categories.")

    st.markdown("### Data Visualizations")
    st.write("Explore dataset distributions and model evaluation results.")

with col2:
    st.markdown("### Model Monitoring")
    st.write("Monitor model version, uptime and system status.")

    st.markdown("### Upload and Retrain")
    st.write("Upload labelled images and initiate model retraining.")

st.divider()

st.caption("TrashNet MLOps Project | Machine Learning Model Lifecycle")
