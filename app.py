import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Auto Insurance Fraud Detection", layout="wide")
st.title("🧠 Auto Insurance Fraud Detection (Inference)")
st.markdown("Upload an unlabeled test file and get predictions for `Fraud_Ind` using the trained model.")

# --- Load model and training columns
MODEL_PATH = r"C:\Users\adita\OneDrive\Desktop\Learnathon 4.0\Learnathon-4.0\best_model.pkl"
COLS_PATH = r"C:\Users\adita\OneDrive\Desktop\Learnathon 4.0\Learnathon-4.0\columns.pkl"

try:
    model = joblib.load(MODEL_PATH)
    expected_columns = joblib.load(COLS_PATH)
    st.success("✅ Model and column schema loaded successfully.")
except Exception as e:
    st.error(f"❌ Error loading model or columns: {e}")
    st.stop()

# --- Upload CSV file
uploaded_file = st.file_uploader("📤 Upload test dataset (CSV format only)", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("### 🔍 Preview of Uploaded Data", df.head())

        # Extract original ID for reference
        id_col = "Claim_ID" if "Claim_ID" in df.columns else df.columns[0]
        df_ids = df[[id_col]]

        # Drop non-numeric or irrelevant columns
        X = df.drop(columns=[col for col in df.columns if col not in expected_columns], errors='ignore')

        # Add missing columns
        for col in expected_columns:
            if col not in X.columns:
                X[col] = 0

        # Reorder columns to match training
        X = X[expected_columns]

        # --- Run predictions
        preds = model.predict(X)
        proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None

        # --- Prepare results
        df_results = df.copy()
        df_results["Fraud_Prediction"] = preds
        if proba is not None:
            df_results["Fraud_Probability"] = proba.round(4)

        # --- Display results
        st.success("✅ Prediction complete!")
        st.write("### 📊 Results with Predictions", df_results.head())

        # --- Downloadable output
        csv = df_results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Results as CSV",
            data=csv,
            file_name="fraud_predictions_results.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"❌ Failed to process and predict: {e}")
