import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from itertools import combinations_with_replacement
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score, recall_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, PolynomialFeatures, StandardScaler, MinMaxScaler, \
    KBinsDiscretizer
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE


def train_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    sensitivity = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()
    g_mean = (tp / (tp + fn)) * (tn / (tn + fp))
    specificity = tn / (tn + fp)
    avg = (f1 + roc_auc + g_mean) / 3

    list_results = [round(result, 2) for result in
                    [accuracy, roc_auc, g_mean, f1, sensitivity, specificity, precision, avg, tn, fp, fn, tp]]

    return list_results, conf_matrix


def display_results(results):
    accuracy = results[0]
    roc_auc = results[1]
    g_mean = results[2]
    f1 = results[3]
    sensitivity = results[4]
    specificity = results[5]
    precision = results[6]
    avg = results[7]

    print("Accuracy:", accuracy)
    print("ROC AUC Score:", roc_auc)
    print("G-Mean:", g_mean)
    print("F1 Score:", f1)
    print("Sensitivity:", sensitivity)
    print("Specificity:", specificity)
    print("Precision:", precision)
    print("AVG:", avg)


def display_confusion_matrix(conf_matrix):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, cmap="PuBuGn", fmt="d")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()


def balance_data_smote(X, y):
    X_bal, y_bal = SMOTE().fit_resample(X, y)
    return X_bal, y_bal


def balance_data_smotetomek(X, y):
    X_bal, y_bal = SMOTETomek().fit_resample(X, y)
    return X_bal, y_bal


# # Test functions: train_model, display_results, display_confusion_matrix
# data = pd.read_csv('heart_disease_risk.csv')
# X = data.drop('decision', axis=1)
# y = data['decision']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# model = LogisticRegression(solver='liblinear')
#
# results, conf_matrix = train_model(model, X_train, X_test, y_train, y_test)
# display_results(results)
# display_confusion_matrix(conf_matrix)


# Feature engineering
def one_hot_encoding(data, categorical_columns):
    encoder = OneHotEncoder()
    encoded_data = pd.DataFrame(encoder.fit_transform(data[categorical_columns]).toarray())
    encoded_data.columns = [f"{col}_{cat}" for col in categorical_columns for cat in
                            encoder.categories_[categorical_columns.index(col)]]
    data = pd.concat([data.reset_index(drop=True), encoded_data], axis=1)
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


def generate_polynomial_feature_names(data, degree):
    feature_names = []
    n_features = data.shape[1]
    for d in range(1, degree + 1):
        for indices in combinations_with_replacement(range(n_features), d):
            name = "x0"
            if len(indices) > 1:
                name += "**" + str(len(indices))
            for i in range(1, n_features):
                if i in indices:
                    name += " * x" + str(i)
            feature_names.append(name)
    return feature_names


def create_polynomial_features(data, degree):
    polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)
    transformed_data = polynomial_features.fit_transform(data)
    new_feature_names = generate_polynomial_feature_names(data, degree)
    data = pd.DataFrame(transformed_data, columns=new_feature_names)
    return data


def create_interaction_features(data):
    interaction_data = pd.DataFrame(index=data.index)
    for i in range(len(data.columns)):
        for j in range(i + 1, len(data.columns)):
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
