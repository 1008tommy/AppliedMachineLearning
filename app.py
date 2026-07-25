import streamlit as st
import pandas as pd
import joblib


# Load the trained model and the exact feature list it expects
model = joblib.load("decision_tree_model.pkl")
feature_names = joblib.load("model_features.pkl")

st.set_page_config(page_title="Network Traffic Classifier", layout="wide")

st.title("Network Traffic Classifier")
st.write(
    "Upload a CSV file containing one or more network flow records. "
    "Each row will be classified as **BENIGN** or **NOT BENIGN**."
)

with st.expander("Required CSV format"):
    st.write(
        f"Your CSV must contain the following {len(feature_names)} columns "
        "(extra columns are ignored, order doesn't matter):"
    )
    st.code(", ".join(feature_names))

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        input_df = pd.read_csv(uploaded_file)
    except Exception as error:
        st.error(f"Could not read the uploaded file: {error}")
        st.stop()

    # Standardise column names in case of stray whitespace
    input_df.columns = input_df.columns.str.strip()

    # Check for missing required columns
    missing_columns = [
        feature for feature in feature_names if feature not in input_df.columns
    ]

    if missing_columns:
        st.error(
            "The uploaded file is missing the following required columns:\n\n"
            + ", ".join(missing_columns)
        )
        st.stop()

    # Keep a copy of any extra identifying columns (if present) for display purposes
    display_columns = [
        column for column in input_df.columns if column not in feature_names
    ]

    # Build the model input using only the expected features, in the correct order
    model_input = input_df[feature_names]

    # Check for missing values in the required columns
    if model_input.isnull().any().any():
        st.warning(
            "Some rows contain missing values in the required feature columns. "
            "These rows will be dropped before prediction."
        )
        valid_mask = ~model_input.isnull().any(axis=1)
        model_input = model_input.loc[valid_mask].reset_index(drop=True)
        input_df = input_df.loc[valid_mask].reset_index(drop=True)

    st.success(f"File uploaded successfully: {len(model_input):,} records ready for prediction.")

    if st.button("Run Predictions", type="primary"):
        predictions = model.predict(model_input)

        results_df = input_df[display_columns].copy() if display_columns else pd.DataFrame(index=input_df.index)
        results_df["Prediction"] = predictions

        # Add class probabilities if the model supports it
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(model_input)
            for class_index, class_label in enumerate(model.classes_):
                results_df[f"P({class_label})"] = probabilities[:, class_index]

        st.subheader("Prediction Results")
        st.dataframe(results_df, use_container_width=True)

        # Summary counts
        st.subheader("Summary")
        summary_counts = pd.Series(predictions).value_counts().rename_axis("Prediction").reset_index(name="Count")
        st.dataframe(summary_counts, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Records", f"{len(predictions):,}")
        with col2:
            not_benign_count = int((predictions == "NOT BENIGN").sum())
            st.metric("Flagged as NOT BENIGN", f"{not_benign_count:,}")

        # Allow downloading the results
        csv_output = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Results as CSV",
            data=csv_output,
            file_name="predictions.csv",
            mime="text/csv"
        )
else:
    st.info("Upload a CSV file above to get started.")