import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load pre-trained models and encoders
scaler = joblib.load("scaler.pkl")
le_gender = joblib.load("label_encoder_gender.pkl")
le_diabetic = joblib.load("label_encoder_diabetic.pkl")
le_smoker = joblib.load("label_encoder_smoker.pkl")
model = joblib.load("best_model1.pkl")


# Helper: encode values using the loaded LabelEncoder, with safe fallbacks
def encode_with_fallback(le, val, fallback_map=None):
    """Try to transform `val` with label encoder `le`.
    If that fails (encoder expects numeric classes), use `fallback_map`.
    The function also tries a case-insensitive match against `le.classes_`.
    """
    # direct transform attempt
    try:
        return int(le.transform([val])[0])
    except Exception:
        pass

    # case-insensitive match against encoder classes
    try:
        for cls in le.classes_:
            if str(cls).lower() == str(val).lower():
                return int(le.transform([cls])[0])
    except Exception:
        pass

    # fallback map (e.g., {'Male':1, 'Female':0})
    if fallback_map and val in fallback_map:
        return int(fallback_map[val])

    raise ValueError(f"Unable to encode value {val!r} using encoder and no suitable fallback found.")

# Streamlit UI Configuration
st.set_page_config(page_title="Insurance Claim Predictor", layout="centered")
st.title(" Health Insurance Payment Prediction App ")
st.write("Enter your health and lifestyle details below to estimate your expected insurance payment amount.")

# Input Form
with st.form("Input Form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=0, max_value=100, value=30)
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
        children = st.number_input("Number of Children", min_value=0, max_value=8, value=0)
    with col2:
        bloodpressure = st.number_input("Blood Pressure", min_value=60.0, max_value=200.0, value=120.0)
        # Show human-readable options regardless of how the encoders were saved
        gender = st.selectbox("Gender", options=["Male", "Female"])
        diabetic = st.selectbox("Diabetic", options=["Yes", "No"])
        smoker = st.selectbox("Smoker", options=["Yes", "No"])

    submitted = st.form_submit_button("🔍 Predict Payment")

# When the user submits the form
if submitted:
    # Create input DataFrame
    input_data = pd.DataFrame({
        "age": [age],
        "gender": [gender],
        "bmi": [bmi],
        "bloodpressure": [bloodpressure],
        "diabetic": [diabetic],
        "children": [children],
        "smoker": [smoker]
    })

    # Label Encoding for categorical features (use safe fallback mapping when needed)
    # NOTE: fallback maps assume Male=1, Female=0 and Yes=1, No=0. Adjust if your encoders use a different mapping.
    gender_map = {"Male": 1, "Female": 0}
    yesno_map = {"Yes": 1, "No": 0}

    input_data["gender"] = [encode_with_fallback(le_gender, gender, fallback_map=gender_map)]
    input_data["diabetic"] = [encode_with_fallback(le_diabetic, diabetic, fallback_map=yesno_map)]
    input_data["smoker"] = [encode_with_fallback(le_smoker, smoker, fallback_map=yesno_map)]

    # Scaling of numerical features
    num_cols = ["age", "bmi", "bloodpressure", "children"]
    input_data[num_cols] = scaler.transform(input_data[num_cols])

    # Predict using trained model
    prediction = model.predict(input_data)[0]

    # Display the result
    st.success(f"### Estimated Insurance Payment Amount: **Ksh.{prediction:,.2f}**")
