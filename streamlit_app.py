import pickle
import streamlit as st
from streamlit_option_menu import option_menu


st.set_page_config(page_title="Disease Prediction", page_icon="+", layout="wide")

with st.sidebar:
    selected = option_menu('Heart Disease Risk Prediction Models',
                           ['Working models',
                            'Logistic Regression',
                            'Naive Bayes',
                            'Support Vector Machine (SVM)',
                            'K-Nearest Neighbors (KNN)',
                            'Decision Tree',
                            'Random Forest',
                            'Gradient Boosting',
                            'Ada Boost',
                            'XGBoost',
                            'SGBoost',
                            'Cat Boost',
                            'Multilayer Perceptron (MLP/Neural Network)',
                            'Light Gradient Boosting Machine (LGBM)'],
                           icons=['heart'],
                           default_index=0)


logistic_regression = pickle.load(open('Sav_models/logistic_regression.sav', 'rb'))
naive_bayes = pickle.load(open('Sav_models/naive_bayes.sav', 'rb'))
svm = pickle.load(open('Sav_models/svm.sav', 'rb'))
# decision_tree = pickle.load(open('Sav_models/decision_tree.sav', 'rb'))
# random_forest = pickle.load(open('Sav_models/random_forest.sav', 'rb'))
# ada_boost = pickle.load(open('Sav_models/ada_boost.sav', 'rb'))
# xgboost = pickle.load(open('Sav_models/xgboost.sav', 'rb'))
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
# elif selected == "XGBoost":
#     if st.button("Predict"):
#         features = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
#         prediction = xgboost.predict(features)
#         if prediction[0] == 1:
#             result = "The person has a risk of heart disease."
#         else:
#             result = "The person does not have a risk of heart disease."
#         st.success(result)
elif selected == "SGBoost":
    if st.button("Predict"):
        features = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
        prediction = sgboost.predict(features)
        if prediction[0] == 1:
            result = "The person has a risk of heart disease."
        else:
            result = "The person does not have a risk of heart disease."
        st.success(result)
elif selected == "Multilayer Perceptron (MLP/Neural Network)":
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
elif selected == "Light Gradient Boosting Machine (LGBM)":
    if st.button("Predict"):
        features = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
        prediction = lgbm.predict(features)
        if prediction[0] == 1:
            result = "The person has a risk of heart disease."
        else:
            result = "The person does not have a risk of heart disease."
        st.success(result)
elif selected == "Working models":
    if st.button("Predict"):
        features = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
        prediction_logistic_regression = logistic_regression.predict(features)
        prediction_naive_bayes = naive_bayes.predict(features)
        prediction_svm = svm.predict(features)
        prediction_sgboost = sgboost.predict(features)
        prediction_cat_boost = cat_boost.predict(features)
        prediction_neural_network = neural_network.predict(features)
        prediction_lgbm = lgbm.predict(features)
        if prediction_logistic_regression[0] == 1:
            result_logistic_regression = "The person has a risk of heart disease. "
        else:
            result_logistic_regression = "The person does not have a risk of heart disease. "
        if prediction_naive_bayes[0] == 1:
            result_naive_bayes = "The person has a risk of heart disease. "
        else:
            result_naive_bayes = "The person does not have a risk of heart disease. "
        if prediction_svm[0] == 1:
            result_svm = "The person has a risk of heart disease. "
        else:
            result_svm = "The person does not have a risk of heart disease. "
        if prediction_sgboost[0] == 1:
            result_sgboost = "The person has a risk of heart disease. "
        else:
            result_sgboost = "The person does not have a risk of heart disease. "
        if prediction_cat_boost[0] == 1:
            result_cat_boost = "The person has a risk of heart disease. "
        else:
            result_cat_boost = "The person does not have a risk of heart disease. "
        if prediction_neural_network[0] == 1:
            result_neural_network = "The person has a risk of heart disease. "
        else:
            result_neural_network = "The person does not have a risk of heart disease. "
        if prediction_lgbm[0] == 1:
            result_lgbm = "The person has a risk of heart disease. "
        else:
            result_lgbm = "The person does not have a risk of heart disease. "
        st.write("Logistic Regression:")
        st.success(result_logistic_regression)
        st.write("Naive Bayes:")
        st.success(result_naive_bayes)
        st.write("Support Vector Machine (SVM):")
        st.success(result_svm)
        st.write("SGBoost:")
        st.success(result_sgboost)
        st.write("Cat Boost:")
        st.success(result_cat_boost)
        st.write("Multilayer Perceptron (MLP/Neural Network):")
        st.success(result_neural_network)
        st.write("Light Gradient Boosting Machine (LGBM):")
        st.success(result_lgbm)
else:
    st.write("Model unavailable")
