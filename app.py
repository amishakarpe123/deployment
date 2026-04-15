import streamlit as st
import pandas as pd
import pickle
import requests
from streamlit_lottie import st_lottie

# --- CONFIGURATION ---
st.set_page_config(page_title="Student Performance Predictor", layout="centered")

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Custom CSS for styling and animations
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.02);
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD ASSETS ---
lottie_coding = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_fcfjwiyb.json")
model = pickle.load(open("model (1).pkl", "rb"))

# --- HEADER ---
with st.container():
    st.title("🎓 Student Performance Predictor")
    if lottie_coding:
        st_lottie(lottie_coding, height=200, key="coding")
    st.write("Enter the details below to predict the student's classification.")

# --- INPUT SECTION ---
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        attendance_rate = st.slider("Attendance Rate (%)", 0, 100, 85)
        study_hours_week = st.number_input("Study Hours Per Week", 0, 168, 20)
        prev_grade = st.number_input("Previous Grade", 0, 100, 75)
        
    with col2:
        extra_curr = st.selectbox("Extracurricular Activities", ["Yes", "No"])
        parent_support = st.selectbox("Parental Support", ["Low", "Medium", "High"])
        final_grade = st.number_input("Current Final Grade", 0, 100, 70)
        study_hours_total = st.number_input("Total Study Hours", 0, 5000, 100)
        attendance_raw = st.number_input("Attendance (Raw Value)", 0, 100, 80)

    submit = st.form_submit_button("✨ Predict Performance")

# --- PREDICTION LOGIC ---
if submit:
    # Pre-processing inputs to match model expectations
    # Note: Ensure encoding (0/1) matches how you trained your model
    input_data = pd.DataFrame({
        'Gender': [1 if gender == "Male" else 0],
        'AttendanceRate': [attendance_rate],
        'StudyHoursPerWeek': [study_hours_week],
        'PreviousGrade': [prev_grade],
        'ExtracurricularActivities': [1 if extra_curr == "Yes" else 0],
        'ParentalSupport': [2 if parent_support == "High" else 1 if parent_support == "Medium" else 0],
        'FinalGrade': [final_grade],
        'Study Hours': [study_hours_total],
        'Attendance (%)': [attendance_raw]
    })

    prediction = model.predict(input_data)
    
    st.balloons()
    st.success(f"### Result: The predicted category is {prediction[0]}")
