import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load model and encoders
model = joblib.load("model/model.pkl")
encoders = joblib.load("model/encoders.pkl")

st.set_page_config("Employee Salary Predictor", layout="wide", page_icon="💼")

st.markdown("""
    <style>
    .css-1n76uvr .stSlider > div > div {
        background-color: #4a90e2 !important;  /* slider color */
    }
    .stButton>button {
        background-color: #2ecc71;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.title("💼 Employee Salary Predictor")
st.subheader("Estimate your monthly salary based on your professional profile")

# User Inputs
with st.form("predict_form"):
    st.markdown("### 👤 Personal & Job Details")
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 65, 30)
        education = st.selectbox("Education Level", {
            8: "10th Pass", 10: "12th Pass", 11: "Diploma", 13: "Bachelor's",
            14: "Master's", 15: "MBA", 16: "PhD", 17: "PostDoc"
        }.values())
        education_num = int([k for k, v in {
            8: "10th Pass", 10: "12th Pass", 11: "Diploma", 13: "Bachelor's",
            14: "Master's", 15: "MBA", 16: "PhD", 17: "PostDoc"
        }.items() if v == education][0])
        hours = st.slider("Hours Worked Per Week", 20, 80, 40)
        language_skills = st.slider("Languages Known", 1, 5, 2)

    with col2:
        experience = st.number_input("Work Experience (years)", 0, 50, 2)
        company_type = st.selectbox("Company Type", encoders['company_type'].classes_)
        job_role = st.selectbox("Job Role", encoders['job_role'].classes_)
        job_level = st.selectbox("Job Level", encoders['job_level'].classes_)
        location = st.selectbox("Location", encoders['location'].classes_)
        remote = st.radio("Remote Work?", ["Yes", "No"])
        gender = st.selectbox("Gender", encoders['gender'].classes_)

    submitted = st.form_submit_button("🔮 Predict Salary")

if submitted:
    # Encode categorical inputs
    def encode(col, val):
        return encoders[col].transform([val])[0]

    input_df = pd.DataFrame([[
        age,
        education_num,
        hours,
        experience,
        encode('company_type', company_type),
        encode('job_role', job_role),
        encode('location', location),
        encode('remote_work', remote),
        encode('job_level', job_level),
        language_skills,
        encode('gender', gender)
    ]], columns=[
        'age', 'education-num', 'hours-per-week', 'work_experience',
        'company_type', 'job_role', 'location', 'remote_work',
        'job_level', 'language_skills', 'gender'
    ])

    # Predict
    predicted_salary = model.predict(input_df)[0]
    tax = 0
    if predicted_salary > 62000:
        tax = predicted_salary * 0.2
    in_hand = predicted_salary - tax

    # Results
    st.markdown("### 💰 Salary Estimate")
    st.success(f"**Total Estimated Salary:** ₹{predicted_salary:,.0f}")
    if tax > 0:
        st.info(f"**Tax Deduction (20%)**: ₹{tax:,.0f}")
    st.success(f"**In-Hand Salary:** ₹{in_hand:,.0f}")

    # 📊 Graphs
    st.markdown("---")
    st.subheader("📊 Salary Insights")
    colA, colB = st.columns(2)

    with colA:
        st.markdown("**📈 Salary vs Experience**")
        exp_range = np.arange(1, 21)
        exp_preds = [
            model.predict(input_df.assign(work_experience=exp))[0] for exp in exp_range
        ]
        fig1, ax1 = plt.subplots()
        ax1.plot(exp_range, exp_preds, marker='o', color='green')
        ax1.set_xlabel("Years of Experience")
        ax1.set_ylabel("Predicted Salary (₹)")
        ax1.set_title("Salary vs Experience")
        st.pyplot(fig1)

    with colB:
        st.markdown("**📊 Market Salary Trends**")
        industries = ["IT", Finance := "Finance", "Healthcare", "Education", "Retail"]
        avg_salaries = [85000, 72000, 66000, 55000, 50000]
        fig2, ax2 = plt.subplots()
        bars = ax2.bar(industries, avg_salaries, color="#1f77b4")
        ax2.bar_label(bars, fmt="₹%d", padding=3)
        ax2.set_ylabel("Average Salary (₹)")
        ax2.set_title("Industry-wise Salary Trends")
        st.pyplot(fig2)
