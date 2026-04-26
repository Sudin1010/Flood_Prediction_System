import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(
    page_title="Flood Risk Prediction",
    page_icon="🌊",
    layout="wide"
)

st.markdown("""
<style>
.main-title {
    font-size: 44px;
    font-weight: 800;
    text-align: center;
}
.sub-title {
    font-size: 20px;
    text-align: center;
    color: #6c757d;
    margin-bottom: 35px;
}
.result-box {
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    font-size: 30px;
    font-weight: 800;
}
.low-risk {
    background-color: #e6f7ec;
    color: #087f23;
}
.medium-risk {
    background-color: #fff4cc;
    color: #8a6500;
}
.high-risk {
    background-color: #ffe5e5;
    color: #b00020;
}
div.stButton > button {
    font-size: 22px;
    font-weight: 700;
    height: 60px;
    border-radius: 14px;
}
</style>
""", unsafe_allow_html=True)

# Load model
with open("models/logistic_regression_scratch.pkl", "rb") as file:
    model_data = pickle.load(file)

weights = model_data["weights"]
bias = model_data["bias"]

# Load scaler
with open("models/scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

feature_names = list(scaler.feature_names_in_)

# Load transformed clean dataset for correct slider ranges
data_path = "data/flood_dataset_clean.csv"

if not os.path.exists(data_path):
    st.error("data/flood_dataset_clean.csv not found.")
    st.stop()

df_range = pd.read_csv(data_path)

for col in ["id", "FloodProbability", "FloodRisk"]:
    if col in df_range.columns:
        df_range = df_range.drop(columns=[col])

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def get_range(feature):
    if feature in df_range.columns:
        min_v = float(df_range[feature].min())
        max_v = float(df_range[feature].max())
        mean_v = float(df_range[feature].mean())

        if min_v == max_v:
            max_v = min_v + 0.01

        return round(min_v, 3), round(max_v, 3), round(mean_v, 3)

    return 0.0, 1.0, 0.5

st.markdown(
    '<div class="main-title">🌊 Flood Risk Prediction System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Predicting flood risk using Logistic Regression implemented from scratch</div>',
    unsafe_allow_html=True
)

climate_features = [
    "MonsoonIntensity",
    "ClimateChange"
]

infrastructure_features = [
    "DamsQuality",
    "DrainageSystems",
    "RiverManagement"
]

environment_features = [
    "Deforestation",
    "Landslides",
    "TopographyDrainage",
    "Watersheds",
    "Siltation",
    "DeterioratingInfrastructure",
    "PopulationScore",
    "InactiveDisasterPreparedness"
]

input_data = {}

def feature_slider(feature):
    min_v, max_v, mean_v = get_range(feature)

    return st.slider(
        label=feature,
        min_value=min_v,
        max_value=max_v,
        value=mean_v,
        step=0.001
    )

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("## 🌧 Climate Factors")
    for feature in climate_features:
        if feature in feature_names:
            input_data[feature] = feature_slider(feature)

with col2:
    st.markdown("## 🏗 Infrastructure Factors")
    for feature in infrastructure_features:
        if feature in feature_names:
            input_data[feature] = feature_slider(feature)

with col3:
    st.markdown("## 🌿 Environmental Factors")
    for feature in environment_features:
        if feature in feature_names:
            input_data[feature] = feature_slider(feature)

# Hidden features use mean value
for feature in feature_names:
    if feature not in input_data:
        _, _, mean_v = get_range(feature)
        input_data[feature] = mean_v

st.markdown("---")

if st.button("Predict Flood Risk", use_container_width=True):
    input_df = pd.DataFrame([input_data])
    input_df = input_df[feature_names]

    # Scale using same scaler used in preprocessing
    input_scaled = scaler.transform(input_df)

    linear_value = np.dot(input_scaled, weights) + bias
    probability = sigmoid(linear_value)[0]

    if probability < 0.35:
        risk_label = "🟢 Low Flood Risk"
        risk_class = "low-risk"
        message = "Low flood risk condition. Regular monitoring is recommended."

    elif probability < 0.70:
        risk_label = "🟡 Medium Flood Risk"
        risk_class = "medium-risk"
        message = "Moderate flood risk. Preventive planning and drainage monitoring are suggested."

    else:
        risk_label = "🔴 High Flood Risk"
        risk_class = "high-risk"
        message = "High flood risk condition. Immediate preparedness is recommended."

    result_col1, result_col2 = st.columns([2, 1])

    with result_col1:
        st.markdown(
            f'<div class="result-box {risk_class}">{risk_label}</div>',
            unsafe_allow_html=True
        )

    with result_col2:
        st.metric("Flood Risk Probability", f"{probability * 100:.2f}%")
        st.progress(float(probability))

    if probability < 0.35:
        st.info(message)
    elif probability < 0.70:
        st.warning(message)
    else:
        st.error(message)

    st.caption(
        "Prediction is based on the combined effect of all transformed features. "
        "Values are scaled internally before being passed to the scratch logistic regression model."
    )