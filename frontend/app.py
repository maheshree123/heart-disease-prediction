import streamlit as st
import joblib
import numpy as np

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

# -------------------------------
# LOAD MODEL
# -------------------------------
model = joblib.load("model/model.pkl")

# -------------------------------
# TITLE
# -------------------------------
st.title("❤️ Heart Disease Prediction")

st.markdown("### Enter Patient Details")

# -------------------------------
# INPUTS
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=0, max_value=120, value=25)
    sex = st.selectbox("Sex (Female=0, Male=1)", [0, 1])
    cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", value=120)
    chol = st.number_input("Cholesterol", value=200)
    fbs = st.selectbox("Fasting Blood Sugar (0/1)", [0, 1])

with col2:
    restecg = st.selectbox("Resting ECG (0-2)", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate", value=150)
    exang = st.selectbox("Exercise Angina (0/1)", [0, 1])
    oldpeak = st.number_input("ST Depression", value=1.0)
    slope = st.selectbox("ST Slope (0-2)", [0, 1, 2])
    ca = st.number_input("Major Vessels (0-3)", min_value=0, max_value=3, value=0)
    thal = st.selectbox("Thalassemia (0-3)", [0, 1, 2, 3])

# -------------------------------
# BUTTON
# -------------------------------
predict_btn = st.button("Predict")

# -------------------------------
# PREDICTION
# -------------------------------
if predict_btn:
    data = [
        age, sex, cp, trestbps, chol, fbs,
        restecg, thalach, exang, oldpeak,
        slope, ca, thal
    ]

    data = np.array(data).reshape(1, -1)

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("⚠️ Heart Disease Detected")
    else:
        st.success("✅ No Heart Disease")