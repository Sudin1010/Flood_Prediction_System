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
    font-size: 46px;
    font-weight: 800;
    text-align: center;
}
.sub-title {
    font-size: 22px;
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

# Load scratch logistic regression model
with open("models/logistic_regression_scratch.pkl", "rb") as file:
    model_data = pickle.load(file)

weights = model_data["weights"]
bias = model_data["bias"]

# Load scaler from preprocessing
with open("models/scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

feature_names = list(scaler.feature_names_in_)

# Load dataset for real slider ranges
data_paths = [
    "data/flood_dataset_clean.csv",
    "data/processed_flood_data.csv",
    "data/train.csv"
]

df_range = None

for path in data_paths:
    if os.path.exists(path):
        df_range = pd.read_csv(path)
        break

if df_range is None:
    st.error("Dataset file not found inside data folder.")
    st.stop()

for col in ["id", "FloodProbability", "FloodRisk"]:
    if col in df_range.columns:
        df_range = df_range.drop(columns=[col])

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def get_feature_range(feature):
    if feature in df_range.columns:
        min_value = float(df_range[feature].min())
        max_value = float(df_range[feature].max())
        mean_value = float(df_range[feature].mean())

        if min_value == max_value:
            max_value = min_value + 1.0

        return min_value, max_value, mean_value

    return 0.0, 10.0, 5.0

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

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("## 🌧 Climate Factors")
    for feature in climate_features:
        if feature in feature_names:
            min_v, max_v, mean_v = get_feature_range(feature)
            input_data[feature] = st.slider(
                label=feature,
                min_value=round(min_v, 2),
                max_value=round(max_v, 2),
                value=round(mean_v, 2)
            )

with col2:
    st.markdown("## 🏗 Infrastructure Factors")
    for feature in infrastructure_features:
        if feature in feature_names:
            min_v, max_v, mean_v = get_feature_range(feature)
            input_data[feature] = st.slider(
                label=feature,
                min_value=round(min_v, 2),
                max_value=round(max_v, 2),
                value=round(mean_v, 2)
            )

with col3:
    st.markdown("## 🌿 Environmental Factors")
    for feature in environment_features:
        if feature in feature_names:
            min_v, max_v, mean_v = get_feature_range(feature)
            input_data[feature] = st.slider(
                label=feature,
                min_value=round(min_v, 2),
                max_value=round(max_v, 2),
                value=round(mean_v, 2)
            )

# Hidden features are filled with dataset mean
for feature in feature_names:
    if feature not in input_data:
        _, _, mean_v = get_feature_range(feature)
        input_data[feature] = mean_v

st.markdown("---")

if st.button("Predict Flood Risk", use_container_width=True):
    input_df = pd.DataFrame([input_data])
    input_df = input_df[feature_names]

    input_scaled = scaler.transform(input_df)

    linear_value = np.dot(input_scaled, weights) + bias
    probability = sigmoid(linear_value)[0]

    if probability < 0.40:
        risk_label = "🟢 Low Flood Risk"
        risk_class = "low-risk"
        message = "Low flood risk condition. Regular monitoring is still recommended."

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

    if probability < 0.40:
        st.info(message)
    elif probability < 0.70:
        st.warning(message)
    else:
        st.error(message)