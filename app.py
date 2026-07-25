import streamlit as st
import pandas as pd
import joblib

# Load model and expected feature order
model = joblib.load("decision_tree_model.pkl")
feature_names = joblib.load("model_features.pkl")

st.title("Network Traffic Classifier")
st.write("Enter flow features to predict BENIGN vs NOT BENIGN traffic.")

# Build input fields dynamically for each feature
user_input = {}
for feature in feature_names:
    user_input[feature] = st.number_input(feature, value=0.0)

if st.button("Predict"):
    input_df = pd.DataFrame([user_input])[feature_names]  # enforce correct column order
    prediction = model.predict(input_df)[0]
    st.subheader(f"Prediction: {prediction}")

    # Show class probabilities
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
        st.write(dict(zip(model.classes_, probabilities)))