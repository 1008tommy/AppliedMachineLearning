import streamlit as st
import pandas as pd
import joblib
 
# ---------------------------------------------------------
# Load the trained model, expected feature list, and sample records
# ---------------------------------------------------------
model = joblib.load("decision_tree_model.pkl")
feature_names = joblib.load("model_features.pkl")
sample_data = pd.read_csv("sample_data.csv")
sample_data.columns = sample_data.columns.str.strip()
 
st.set_page_config(page_title="Network Traffic Classifier", layout="wide")
 
st.title("Network Traffic Classifier")
st.write(
    "Pick a sample record below to auto-fill the form, then adjust any values "
    "and click **Predict** to see whether the model classifies it as BENIGN or NOT BENIGN."
)
 
# ---------------------------------------------------------
# Sample record selector
# ---------------------------------------------------------
sample_labels = sample_data["Original Label"].tolist()
sample_options = ["-- Start with default values --"] + [
    f"{i}: {label}" for i, label in enumerate(sample_labels)
]
 
selected_sample = st.selectbox("Load a sample record", sample_options)
 
if selected_sample == "-- Start with default values --":
    default_values = {feature: 0.0 for feature in feature_names}
else:
    sample_index = int(selected_sample.split(":")[0])
    default_values = sample_data.loc[sample_index, feature_names].to_dict()
 
st.divider()
 
# ---------------------------------------------------------
# Input form
# ---------------------------------------------------------
st.subheader("Flow Feature Values")
st.caption("Values are pre-filled from the selected sample. Edit any field before predicting.")
 
with st.form("prediction_form"):
    user_input = {}
 
    # Lay fields out in 3 columns to keep the form compact given the large feature count
    columns = st.columns(3)
    for index, feature in enumerate(feature_names):
        with columns[index % 3]:
            user_input[feature] = st.number_input(
                feature,
                value=float(default_values[feature]),
                format="%.6f"
            )
 
    submitted = st.form_submit_button("Predict", type="primary")
 
# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
if submitted:
    input_df = pd.DataFrame([user_input])[feature_names]
    prediction = model.predict(input_df)[0]
 
    st.divider()
    st.subheader("Prediction Result")
 
    if prediction == "BENIGN":
        st.success(f"Prediction: **{prediction}**")
    else:
        st.error(f"Prediction: **{prediction}**")