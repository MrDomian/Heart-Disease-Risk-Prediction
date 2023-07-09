import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score, recall_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, PolynomialFeatures, StandardScaler, MinMaxScaler, KBinsDiscretizer


def display_results(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    sensitivity = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()
    g_mean = (tp / (tp + fn)) * (tn / (tn + fp))
    specificity = tn / (tn + fp)

    print("Accuracy:", round(accuracy, 2))
    print("ROC AUC Score:", round(roc_auc, 2))
    print("G-Mean:", round(g_mean, 2))
    print("F1 Score:", round(f1, 2))
    print("Sensitivity:", round(sensitivity, 2))
    print("Specificity:", round(specificity, 2))
    print("Precision:", round(precision, 2))
    # print("True Negative:", cm[0, 0])
    # print("False Positive:", cm[0, 1])
    # print("False Negative:", cm[1, 0])
    # print("True Positive:", cm[1, 1])

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap="PuBuGn", fmt="d")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()


# # Test function display_results
# data = pd.read_csv('heart_disease_risk.csv')
# X = data.drop('decision', axis=1)
# y = data['decision']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#
# model = LogisticRegression(solver='liblinear')
# display_results(model, X_train, X_test, y_train, y_test)


def one_hot_encoding(data, categorical_columns):
    encoder = OneHotEncoder()
    encoded_data = pd.DataFrame(encoder.fit_transform(data[categorical_columns]).toarray(), columns=encoder.get_feature_names(categorical_columns))
    data = pd.concat([data, encoded_data], axis=1)
    data.drop(categorical_columns, axis=1, inplace=True)
    return data


def label_encoding(data, categorical_columns):
    encoder = LabelEncoder()
    for col in categorical_columns:
        data[col] = encoder.fit_transform(data[col])
    return data


def ordinal_encoding(data, categorical_columns, ordinal_mapping):
    for col in categorical_columns:
        data[col] = data[col].map(ordinal_mapping[col])
    return data


def create_polynomial_features(data, degree):
    polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)
    transformed_data = polynomial_features.fit_transform(data)
    new_feature_names = polynomial_features.get_feature_names(data.columns)
    data = pd.DataFrame(transformed_data, columns=new_feature_names)
    return data


def create_interaction_features(data):
    interaction_data = pd.DataFrame(index=data.index)
    for i in range(len(data.columns)):
        for j in range(i+1, len(data.columns)):
            interaction_data[data.columns[i] + '*' + data.columns[j]] = data.iloc[:, i] * data.iloc[:, j]
    data = pd.concat([data, interaction_data], axis=1)
    return data


def standardize_features(data):
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(data)
    data = pd.DataFrame(standardized_data, columns=data.columns)
    return data


def normalize_features(data):
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data)
    data = pd.DataFrame(normalized_data, columns=data.columns)
    return data


def discretize_features(data, continuous_columns, n_bins):
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal')
    discretized_data = discretizer.fit_transform(data[continuous_columns])
    discretized_data = pd.DataFrame(discretized_data, columns=continuous_columns)
    data[continuous_columns] = discretized_data
    return data


def remove_highly_correlated_features(data, threshold):
    correlation_matrix = data.corr().abs()
    upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    correlated_features = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
    data.drop(correlated_features, axis=1, inplace=True)
    return data


def remove_features_with_high_missing_data(data, missing_threshold):
    missing_data = data.isnull().mean()
    features_to_remove = missing_data[missing_data > missing_threshold].index
    data.drop(features_to_remove, axis=1, inplace=True)
    return data


def extract_time_features(data, time_column):
    data[time_column] = pd.to_datetime(data[time_column])
    data['year'] = data[time_column].dt.year
    data['month'] = data[time_column].dt.month
    data['day_of_week'] = data[time_column].dt.dayofweek
    data['hour'] = data[time_column].dt.hour
    return data


def create_time_features(data, time_column):
    data[time_column] = pd.to_datetime(data[time_column])
    data['season'] = data[time_column].dt.quarter
    data['is_weekend'] = data[time_column].dt.weekday.isin([5, 6]).astype(int)
    return data


def extract_text_features(data, text_column):
    vectorizer = CountVectorizer()
    text_features = vectorizer.fit_transform(data[text_column])
    text_feature_names = vectorizer.get_feature_names()
    text_features = pd.DataFrame(text_features.toarray(), columns=text_feature_names)
    data = pd.concat([data, text_features], axis=1)
    data.drop(text_column, axis=1, inplace=True)
    return data


def transform_text_to_features(data, text_column):
    vectorizer = TfidfVectorizer()
    text_features = vectorizer.fit_transform(data[text_column])
    text_feature_names = vectorizer.get_feature_names()
    text_features = pd.DataFrame(text_features.toarray(), columns=text_feature_names)
    data = pd.concat([data, text_features], axis=1)
    data.drop(text_column, axis=1, inplace=True)
    return data