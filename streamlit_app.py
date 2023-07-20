import pickle
import streamlit as st
from streamlit_option_menu import option_menu


st.set_page_config(page_title="Disease Prediction", page_icon="+", layout="wide")

with st.sidebar:
    selected = option_menu('Heart Disease Risk Prediction Models',
                           ['Logistic Regression',
                            'Naive Bayes',
                            'Decision Tree',
                            'Random Forest',
                            'K-Nearest Neighbors (KNN)',
                            'Support Vector Machine (SVM)',
                            'Gradient Boosting',
                            'Ada Boost',
                            'Cat Boost',
                            'XGBoost',
                            'SGBoost',
                            'MLP (Neural Network)',
                            'LGBM'],
                           icons=['heart'],
                           default_index=0)


logistic_regression = pickle.load(open('Sav_models/logistic_regression.sav', 'rb'))
naive_bayes = pickle.load(open('Sav_models/naive_bayes.sav', 'rb'))
svm = pickle.load(open('Sav_models/svm.sav', 'rb'))
# decision_tree = pickle.load(open('Sav_models/decision_tree.sav', 'rb'))
# random_forest = pickle.load(open('Sav_models/random_forest.sav', 'rb'))
# ada_boost = pickle.load(open('Sav_models/ada_boost.sav', 'rb'))
xgboost = pickle.load(open('Sav_models/xgboost.sav', 'rb'))
sgboost = pickle.load(open('Sav_models/sgboost.sav', 'rb'))
neural_network = pickle.load(open('Sav_models/neural_network.sav', 'rb'))
# knn = pickle.load(open('Sav_models/knn.sav', 'rb'))
# gradient_boosting = pickle.load(open('Sav_models/gradient_boosting.sav', 'rb'))
cat_boost = pickle.load(open('Sav_models/cat_boost.sav', 'rb'))
lgbm = pickle.load(open('Sav_models/lgbm.sav', 'rb'))


st.title("Heart Disease Risk Prediction")
col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Age")
    trestbps = st.number_input("Resting Blood Pressure")
    restecg = st.selectbox("Resting Electrocardiographic Results", [0, 1, 2])
    oldpeak = st.number_input("ST Depression Induced by Exercise")
    thal = st.selectbox("Thal", [3, 6, 7])
with col2:
    sex = st.selectbox("Sex", [0, 1])
    chol = st.number_input("Serum Cholestoral in mg/dl")
    thalach = st.number_input("Maximum Heart Rate Achieved")
    slope = st.selectbox("Slope of the Peak Exercise ST Segment", [1, 2, 3])
with col3:
    cp = st.selectbox("Chest Pain Type", [1, 2, 3, 4])
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    ca = st.number_input("Major Vessels Colored by Flourosopy")

if selected == "Logistic Regression":
    if st.button("Predict"):
        features = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
        prediction = logistic_regression.predict(features)
        if prediction[0] == 1:
            result = "The person has a risk of heart disease."
        else:
            result = "The person does not have a risk of heart disease."
        st.success(result)
elif selected == "Naive Bayes":
    if st.button("Predict"):
        features = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
        prediction = naive_bayes.predict(features)
        if prediction[0] == 1:
            result = "The person has a risk of heart disease."
        else:
            result = "The person does not have a risk of heart disease."
        st.success(result)
elif selected == "Support Vector Machine (SVM)":
    if st.button("Predict"):
        features = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
        prediction = svm.predict(features)
        if prediction[0] == 1:
            result = "The person has a risk of heart disease."
        else:
            result = "The person does not have a risk of heart disease."
        st.success(result)
# elif selected == "Decision Tree":
#     if st.button("Predict"):
#         features = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
#         prediction = decision_tree.predict(features)
#         if prediction[0] == 1:
#             result = "The person has a risk of heart disease."
#         else:
#             result = "The person does not have a risk of heart disease."
#         st.success(result)
# elif selected == "Random Forest":
#     if st.button("Predict"):
#         features = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
#         prediction = random_forest.predict(features)
#         if prediction[0] == 1:
#             result = "The person has a risk of heart disease."
#         else:
#             result = "The person does not have a risk of heart disease."
#         st.success(result)
# elif selected == "Ada Boost":
#     if st.button("Predict"):
#         features = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
#         prediction = ada_boost.predict(features)
#         if prediction[0] == 1:
#             result = "The person has a risk of heart disease."
#         else:
#             result = "The person does not have a risk of heart disease."
#         st.success(result)
elif selected == "XGBoost":
    if st.button("Predict"):
        features = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
        prediction = xgboost.predict(features)
        if prediction[0] == 1:
            result = "The person has a risk of heart disease."
        else:
            result = "The person does not have a risk of heart disease."
        st.success(result)
elif selected == "SGBoost":
    if st.button("Predict"):
        features = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
        prediction = sgboost.predict(features)
        if prediction[0] == 1:
            result = "The person has a risk of heart disease."
        else:
            result = "The person does not have a risk of heart disease."
        st.success(result)
elif selected == "MLP (Neural Network)":
    if st.button("Predict"):
        features = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
        prediction = neural_network.predict(features)
        if prediction[0] == 1:
            result = "The person has a risk of heart disease."
        else:
            result = "The person does not have a risk of heart disease."
        st.success(result)
# elif selected == "K-Nearest Neighbors (KNN)":
#     if st.button("Predict"):
#         features = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
#         prediction = knn.predict(features)
#         if prediction[0] == 1:
#             result = "The person has a risk of heart disease."
#         else:
#             result = "The person does not have a risk of heart disease."
#         st.success(result)
# elif selected == "Gradient Boosting":
#     if st.button("Predict"):
#         features = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
#         prediction = gradient_boosting.predict(features)
#         if prediction[0] == 1:
#             result = "The person has a risk of heart disease."
#         else:
#             result = "The person does not have a risk of heart disease."
#         st.success(result)
elif selected == "Cat Boost":
    if st.button("Predict"):
        features = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
        prediction = cat_boost.predict(features)
        if prediction[0] == 1:
            result = "The person has a risk of heart disease."
        else:
            result = "The person does not have a risk of heart disease."
        st.success(result)
elif selected == "LGBM":
    if st.button("Predict"):
        features = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
        prediction = lgbm.predict(features)
        if prediction[0] == 1:
            result = "The person has a risk of heart disease."
        else:
            result = "The person does not have a risk of heart disease."
        st.success(result)

else:
    st.title("Other Disease Prediction")
