import joblib
import numpy as np

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

model = joblib.load("model/model.pkl")
import streamlit as st
# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

# -------------------------------
# CUSTOM CSS (PREMIUM UI)
# -------------------------------
st.markdown("""
<style>
.stApp {
    background-color: #0b1220;
}

div[data-baseweb="input"], div[data-baseweb="select"] {
    border-radius: 10px !important;
    border: 1px solid #2c2c2c !important;
}

.stButton>button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 10px;
    height: 45px;
    width: 150px;
    font-size: 16px;
    font-weight: bold;
}

.stButton>button:hover {
    background-color: #ff1f1f;
}

.block-container {
    padding-top: 2rem;
}

.stSuccess, .stError {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.title("🩺 About")
st.sidebar.info("This app predicts heart disease using Machine Learning.")

# -------------------------------
# TITLE
# -------------------------------
st.markdown(
    "<h1 style='text-align:center; color:#ff4b4b;'>❤️ Heart Disease Prediction</h1>",
    unsafe_allow_html=True
)

st.markdown("---")
st.markdown("### 📋 Enter Patient Details")

# -------------------------------
# INPUT LAYOUT (2 COLUMNS)
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=0, max_value=120, value=25)

    sex = st.selectbox("Sex (Female=0, Male=1)", [0, 1])

    cp = st.selectbox(
        "Chest Pain Type (0=Typical,1=Atypical,2=Non-anginal,3=Asymptomatic)",
        [0, 1, 2, 3]
    )

    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", value=120)

    chol = st.number_input("Cholesterol (mg/dl)", value=200)

    fbs = st.selectbox("Fasting Blood Sugar (0=≤120,1=>120)", [0, 1])

with col2:
    restecg = st.selectbox(
        "Resting ECG (0=Normal,1=ST-T abnormality,2=LV hypertrophy)",
        [0, 1, 2]
    )

    thalach = st.number_input("Max Heart Rate", value=150)

    exang = st.selectbox("Exercise Induced Angina (0=No,1=Yes)", [0, 1])

    oldpeak = st.number_input("ST Depression", value=1.0)

    slope = st.selectbox("ST Slope (0=Upsloping,1=Flat,2=Downsloping)", [0, 1, 2])

    ca = st.number_input("Number of Major Vessels (0–3)", min_value=0, max_value=3, value=0)

    thal = st.selectbox(
        "Thalassemia (0=Normal,1=Fixed Defect,2=Reversible,3=Unknown)",
        [0, 1, 2, 3]
    )

# -------------------------------
# BUTTON (CENTERED)
# -------------------------------
st.markdown("<br>", unsafe_allow_html=True)

col_center = st.columns([3,1,3])
with col_center[1]:
    predict_btn = st.button("🔍 Predict")

# -------------------------------
# PREDICTION
# -------------------------------
if predict_btn:
    data = [
        age, sex, cp, trestbps, chol, fbs,
        restecg, thalach, exang, oldpeak,
        slope, ca, thal
    ]

    try:
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json={"data": data}
        )

        result = response.json()

        if "prediction" in result:
            if result["prediction"] == 1:
                st.error("⚠️ Heart Disease Detected")
            else:
                st.success("✅ No Heart Disease")
        else:
            st.warning("⚠️ No result from backend")

    except Exception as e:
        st.error(f"❌ Error: {e}")