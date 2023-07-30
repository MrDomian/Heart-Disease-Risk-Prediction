import pickle
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_option_menu import option_menu


def load_models():
    models = {
        'Logistic Regression': pickle.load(open('Sav_models/logistic_regression.sav', 'rb')),
        'Ridge Classifier': pickle.load(open('Sav_models/ridge.sav', 'rb')),
        'Naive Bayes': pickle.load(open('Sav_models/naive_bayes.sav', 'rb')),
        'Support Vector Machine (SVM)': pickle.load(open('Sav_models/svm.sav', 'rb')),
        'K-Nearest Neighbors (KNN)': pickle.load(open('Sav_models/knn.sav', 'rb')),
        'Nearest Centroid': pickle.load(open('Sav_models/nearest_centroid.sav', 'rb')),
        'Linear Discriminant': pickle.load(open('Sav_models/linear_discriminant.sav', 'rb')),
        'Quadratic Discriminant': pickle.load(open('Sav_models/quadratic_discriminant.sav', 'rb')),
        'Decision Tree': pickle.load(open('Sav_models/decision_tree.sav', 'rb')),
        'Extra Trees': pickle.load(open('Sav_models/extra_trees.sav', 'rb')),
        'Random Forest': pickle.load(open('Sav_models/random_forest.sav', 'rb')),
        'Gradient Boosting': pickle.load(open('Sav_models/gradient_boosting.sav', 'rb')),
        'SGBoost': pickle.load(open('Sav_models/sgboost.sav', 'rb')),
        # 'XGBoost': pickle.load(open('Sav_models/xgboost.sav', 'rb')), # error
        'Ada Boost': pickle.load(open('Sav_models/ada_boost.sav', 'rb')),
        'Cat Boost': pickle.load(open('Sav_models/cat_boost.sav', 'rb')),
        'Light Gradient Boosting Machine (LGBM)': pickle.load(open('Sav_models/lgbm.sav', 'rb')),
        'Multilayer Perceptron (MLP/Neural Network)': pickle.load(open('Sav_models/neural_network.sav', 'rb')),
    }
    return models


def predict_risk(models, selected_model, features):
    model = models[selected_model]
    prediction = model.predict(features)
    return prediction[0]


def show_user_inputs():
    st.title("Heart Disease Risk Prediction")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", step=1, min_value=1)
        trestbps = st.number_input("Resting Blood Pressure", step=1, min_value=1)
        restecg = st.selectbox("Resting Electrocardiographic Results", [0, 1, 2])
        oldpeak = st.number_input("ST Depression Induced by Exercise")
        thal = st.selectbox("Thal", [3, 6, 7])

    with col2:
        sex = st.selectbox("Sex", [0, 1])
        chol = st.number_input("Serum Cholestoral in mg/dl", step=1, min_value=1)
        thalach = st.number_input("Maximum Heart Rate Achieved", step=1, min_value=1)
        slope = st.selectbox("Slope of the Peak Exercise ST Segment", [1, 2, 3])

    with col3:
        cp = st.selectbox("Chest Pain Type", [1, 2, 3, 4])
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
        exang = st.selectbox("Exercise Induced Angina", [0, 1])
        ca = st.selectbox("Major Vessels Colored by Flourosopy", [0, 1, 2, 3])

    return [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]


def main():
    st.set_page_config(page_title="Disease Prediction", page_icon="+", layout="wide")
    with st.sidebar:
        models_list = ['Working models',
                       'Logistic Regression',
                       'Ridge Classifier',
                       'Naive Bayes',
                       'Support Vector Machine (SVM)',
                       'K-Nearest Neighbors (KNN)',
                       'Nearest Centroid',
                       'Linear Discriminant',
                       'Quadratic Discriminant',
                       'Decision Tree',
                       'Extra Trees',
                       'Random Forest',
                       'Gradient Boosting',
                       'SGBoost',
                       # 'XGBoost',
                       'Ada Boost',
                       'Cat Boost',
                       'Light Gradient Boosting Machine (LGBM)',
                       'Multilayer Perceptron (MLP/Neural Network)']

        selected_model = option_menu('Select a model', models_list, icons=['heart'], default_index=0)
    models = load_models()

    if selected_model == 'Working models':
        features = show_user_inputs()
        if st.button("Predict"):
            results = {}
            for model_name in models:
                result = predict_risk(models, model_name, [features])
                results[model_name] = result

            results_df = pd.DataFrame(results.items(), columns=['Model', 'Result'])
            results_df['Result'] = results_df['Result'].astype(str)

            fig = px.bar(results_df, x='Model', y='Result', color='Result',
                         labels={'Model': 'Machine learning model', 'Result': 'Heart disease risk predict'},
                         color_discrete_map={'1': 'green'},
                         category_orders={'Model': models_list[1:]})
            fig.update_layout(xaxis_tickangle=-30, xaxis_tickfont=dict(size=14))
            fig.update_traces(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            for model_name, result in results.items():
                st.write(model_name + ":")
                if result == 1:
                    st.success("The person has a risk of heart disease.")
                else:
                    st.success("The person does not have a risk of heart disease.")
    else:
        if selected_model in models:
            features = show_user_inputs()

            if st.button("Predict"):
                result = predict_risk(models, selected_model, [features])
                st.write(selected_model + ":")
                if result == 1:
                    st.success("The person has a risk of heart disease.")
                else:
                    st.success("The person does not have a risk of heart disease.")
        else:
            st.write("Model unavailable")


if __name__ == "__main__":
    main()
