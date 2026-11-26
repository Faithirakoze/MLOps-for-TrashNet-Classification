import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Data Visualizations | TrashNet",
    layout="wide"
)

st.title("Data Visualizations")
st.write(
    "Explore dataset distribution and evaluate "
    "the performance of the TrashNet classification model."
)

# --------------------------------------------------
# DATASET DISTRIBUTION
# --------------------------------------------------

st.header("Dataset Distribution")

class_names = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

# Original TrashNet dataset class counts
class_counts = {
    "cardboard": 403,
    "glass": 501,
    "metal": 410,
    "paper": 594,
    "plastic": 482,
    "trash": 137
}

distribution_df = pd.DataFrame({
    "Category": list(class_counts.keys()),
    "Images": list(class_counts.values())
})

col1, col2 = st.columns(2)

with col1:
    st.subheader("Images per Category")

    fig = px.bar(
        distribution_df,
        x="Category",
        y="Images",
        color="Category",
        text="Images",
        title="TrashNet Class Distribution"
    )

    fig.update_layout(showlegend=False)

    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Dataset Composition")

    fig = px.pie(
        distribution_df,
        names="Category",
        values="Images",
        hole=0.4,
        title="Percentage of Images per Category"
    )

    st.plotly_chart(fig, use_container_width=True)

st.divider()

# --------------------------------------------------
# HELD-OUT TEST CONFUSION MATRIX
# --------------------------------------------------

st.header("Model Evaluation")
st.subheader("Held-out Test Confusion Matrix")

# Rows = actual classes
# Columns = predicted classes

confusion_matrix = [
    [71, 0, 1, 7, 1, 1],
    [0, 83, 9, 1, 7, 0],
    [0, 3, 71, 2, 4, 2],
    [4, 0, 0, 110, 3, 2],
    [0, 9, 6, 2, 75, 5],
    [1, 0, 2, 5, 2, 17]
]

# Create confusion matrix DataFrame
confusion_df = pd.DataFrame(
    confusion_matrix,
    index=class_names,
    columns=class_names
)

fig = px.imshow(
    confusion_df,
    text_auto=True,
    color_continuous_scale="Blues",
    labels={
        "x": "Predicted Category",
        "y": "Actual Category",
        "color": "Number of Images"
    },
    title="Held-out Test Confusion Matrix",
    aspect="equal"
)

fig.update_layout(
    height=600,
    xaxis_side="bottom"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# --------------------------------------------------
# OVERALL PERFORMANCE METRICS
# --------------------------------------------------

st.header("Overall Model Performance")

metrics = {
    "Accuracy": 0.8439,
    "Macro Precision": 0.8194,
    "Macro Recall": 0.8166,
    "Macro F1-score": 0.8173
}

col1, col2, col3, col4 = st.columns(4)

for column, (name, value) in zip(
    [col1, col2, col3, col4],
    metrics.items()
):
    with column:
        st.metric(
            name,
            f"{value * 100:.2f}%"
        )

# --------------------------------------------------
# PER-CLASS PERFORMANCE
# --------------------------------------------------

st.divider()
st.header("Per-Class Performance")

performance_data = {
    "Category": class_names,
    "Precision": [0.93, 0.87, 0.80, 0.87, 0.82, 0.63],
    "Recall": [0.88, 0.83, 0.87, 0.92, 0.77, 0.63],
    "F1-score": [0.90, 0.85, 0.83, 0.89, 0.79, 0.63],
    "Support": [81, 100, 82, 119, 97, 27]
}

performance_df = pd.DataFrame(performance_data)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Precision, Recall and F1-score")

    performance_long = performance_df.melt(
        id_vars="Category",
        value_vars=["Precision", "Recall", "F1-score"],
        var_name="Metric",
        value_name="Score"
    )

    fig = px.bar(
        performance_long,
        x="Category",
        y="Score",
        color="Metric",
        barmode="group",
        title="Per-Class Evaluation Metrics",
        range_y=[0, 1]
    )

    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Test Images per Category")

    fig = px.bar(
        performance_df,
        x="Category",
        y="Support",
        color="Category",
        text="Support",
        title="Held-out Test Support"
    )

    fig.update_layout(showlegend=False)

    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# PERFORMANCE TABLE
# --------------------------------------------------

st.subheader("Detailed Classification Report")

display_df = performance_df.copy()

for column in ["Precision", "Recall", "F1-score"]:
    display_df[column] = (
        display_df[column] * 100
    ).round(2)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)

st.divider()

# --------------------------------------------------
# OBSERVATIONS
# --------------------------------------------------

st.header("Model Evaluation Insights")

st.markdown("""
- **Paper** achieved the highest recall at 92%.
- **Cardboard** achieved the highest F1-score at 90%.
- **Trash** had the lowest F1-score at 63%.
- Some glass images were misclassified as metal or plastic.
- The model achieved 84.39% overall test accuracy.

These results highlight areas where additional training data
and model improvements may be beneficial.
""")

st.caption(
    "TrashNet MLOps | Model Evaluation Dashboard"
)
