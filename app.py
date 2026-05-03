import streamlit as st
import pandas as pd

# Page config
st.set_page_config(page_title="Student Depression Predictor", page_icon="🧠", layout="wide")

# 🔥 MODEL LOAD FIX (NO PICKLE ISSUE)
import train_model

model, le = train_model.get_model()

# Title
st.title("Student Depression Predictor")
st.write("Fill the details below to predict whether a student may be at risk of depression.")

# Layout
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=10, max_value=100, value=20)
    gender = st.selectbox("Gender", ["Male", "Female"])
    academic_pressure = st.slider("Academic Pressure", 1, 5, 3)
    work_pressure = st.slider("Work Pressure", 1, 5, 3)
    cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=7.5, step=0.1)
    study_sat = st.slider("Study Satisfaction", 1, 5, 3)
    job_sat = st.slider("Job Satisfaction", 1, 5, 3)

with col2:
    sleep = st.selectbox(
        "Sleep Duration",
        ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"]
    )
    diet = st.selectbox("Dietary Habits", ["Healthy", "Moderate", "Unhealthy"])
    degree = st.text_input("Degree")
    suicidal = st.selectbox("Have you ever had suicidal thoughts ?", ["Yes", "No"])
    hours = st.number_input("Work/Study Hours", min_value=0, max_value=24, value=8)
    financial = st.slider("Financial Stress", 1, 5, 3)
    family_hist = st.selectbox("Family History of Mental Illness", ["Yes", "No"])
    profession = st.text_input("Profession", value="Student")

# Predict button
if st.button("Predict Depression"):
    input_data = pd.DataFrame({
        "Gender": [gender],
        "Age": [age],
        "Profession": [profession],
        "Academic Pressure": [academic_pressure],
        "Work Pressure": [work_pressure],
        "CGPA": [cgpa],
        "Study Satisfaction": [study_sat],
        "Job Satisfaction": [job_sat],
        "Sleep Duration": [sleep],
        "Dietary Habits": [diet],
        "Degree": [degree],
        "Have you ever had suicidal thoughts 0": [1 if suicidal == "Yes" else 0],
        "Work/Study Hours": [hours],
        "Financial Stress": [financial],
        "Family History of Mental Illness": [family_hist]
    })

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("Prediction: Depressed")

        st.warning("Need Someone to Talk To?")
        st.info("""
This prediction is only an AI-based estimate, not a medical diagnosis.

Please consider these support options:
- Talk to a trusted friend, family member, or teacher.
- Reach out to your college counseling center.
- Kiran Mental Health Helpline (India): 1800-599-0019
- AASRA Helpline: +91 22 2754 6669
- Vandrevala Foundation Helpline: 9999 666 555
        """)
    else:
        st.success("Prediction: Not Depressed")
        st.info("You are showing positive indicators. Keep taking care of your mental well-being.")