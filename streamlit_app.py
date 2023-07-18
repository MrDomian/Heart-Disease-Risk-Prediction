import pickle
import streamlit as st


st.set_page_config(page_title="Disease Prediction", page_icon="+")
hide_st_style = """
<style>
    footer {visibility: hidden;}
</style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

model = pickle.load(open('model.sav', 'rb'))

with st.sidebar:
    selected = st.selectbox("Choose Disease", ["Heart Disease"])

if selected == "Heart Disease":
    st.title("Heart Disease Risk Prediction")
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age")
        sex = st.selectbox("Sex", [0, 1])
        cp = st.selectbox("Chest Pain Type", [1, 2, 3, 4])
        trestbps = st.number_input("Resting Blood Pressure")
        chol = st.number_input("Serum Cholestoral in mg/dl")
    with col2:
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
        restecg = st.selectbox("Resting Electrocardiographic Results", [0, 1, 2])
        thalach = st.number_input("Maximum Heart Rate Achieved")
        exang = st.selectbox("Exercise Induced Angina", [0, 1])
    with col3:
        oldpeak = st.number_input("ST Depression Induced by Exercise")
        slope = st.selectbox("Slope of the Peak Exercise ST Segment", [1, 2, 3])
        ca = st.number_input("Major Vessels Colored by Flourosopy")
        thal = st.selectbox("Thal", [3, 6, 7])

    if st.button("Predict"):
        features = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
        prediction = model.predict(features)
        if prediction[0] == 1:
            result = "The person has a risk of heart disease."
        else:
            result = "The person does not have a risk of heart disease."
        st.success(result)

else:
    st.title("Other Disease Prediction")