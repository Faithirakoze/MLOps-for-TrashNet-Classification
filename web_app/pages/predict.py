import os
import streamlit as st
import requests
import pandas as pd
import plotly.express as px

# Configuration
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
PREDICT_URL = f"{API_URL}/predict"

st.set_page_config(
    page_title="Waste Prediction | TrashNet",
    page_icon="♻️",
    layout="wide"
)

st.title("♻️ Waste Classification")
st.write("Upload an image to identify its waste category.")

st.divider()

# Image upload
uploaded_file = st.file_uploader(
    "Choose a waste image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        st.image(
            uploaded_file,
            caption=uploaded_file.name,
            use_container_width=True
        )

    with col2:
        st.subheader("Prediction")

        if st.button("Classify Image", type="primary"):

            with st.spinner("Analyzing image..."):

                try:
                    uploaded_file.seek(0)

                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type
                        )
                    }

                    response = requests.post(
                        PREDICT_URL,
                        files=files,
                        timeout=60
                    )

                    if response.status_code == 200:

                        result = response.json()

                        predicted_class = result["predicted_class"]
                        confidence = result["confidence"]
                        probabilities = result["probabilities"]

                        st.success("Classification completed!")

                        st.markdown("### Predicted Category")

                        st.header(predicted_class.capitalize())

                        st.metric(
                            label="Confidence",
                            value=f"{confidence * 100:.2f}%"
                        )

                        st.progress(float(confidence))

                        # Probability distribution
                        st.subheader("Class Probabilities")

                        df = pd.DataFrame({
                            "Category": list(probabilities.keys()),
                            "Probability": [
                                value * 100
                                for value in probabilities.values()
                            ]
                        })

                        df = df.sort_values(
                            by="Probability",
                            ascending=True
                        )

                        fig = px.bar(
                            df,
                            x="Probability",
                            y="Category",
                            orientation="h",
                            title="Prediction Probability by Category",
                            labels={
                                "Probability": "Probability (%)",
                                "Category": "Waste Category"
                            }
                        )

                        fig.update_layout(
                            xaxis_range=[0, 100],
                            showlegend=False
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True
                        )

                    else:
                        st.error(
                            f"Prediction failed: {response.text}"
                        )

                except requests.exceptions.RequestException as error:
                    st.error(f"Could not connect to the API: {error}")

                except (KeyError, ValueError) as error:
                    st.error(f"Unexpected API response: {error}")
