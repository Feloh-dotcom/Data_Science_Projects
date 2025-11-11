import streamlit as st
import numpy as np
import joblib
import warnings

warnings.filterwarnings("ignore")

st.title("Student Exam Score Predictor")

try:
    model = joblib.load("best_model.pkl")
    st.write("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Input fields below...
Study_Hours = st.slider("Study Hours Per Day", 0.0, 24.0, 8.0)
Attendance = st.slider("Attendance Percentage", 0.0, 100.0, 80.0)
Sleep_Hours = st.slider("Sleep Hours Per Night", 0.0, 24.0, 8.0)
mental_health = st.slider("Mental Health Rating (1-10)", 0, 10, 9)
part_time_job = st.selectbox("Part-Time Job", ["NO", "YES"])

ptj_encoded = 1 if part_time_job == "YES" else 0

if st.button("Predict Exam Score"):
    input_data = np.array([[mental_health, Sleep_Hours, ptj_encoded, Attendance, Study_Hours]])
    prediction = model.predict(input_data)[0]
    prediction1 = max(0, min(100, prediction))
    st.success(f"🎉 Predicted Exam Score: {prediction1:.2f}")
