import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score, recall_score, confusion_matrix


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


# Test function display_results
# data = pd.read_csv('heart_disease_risk.csv')
# X = data.drop('decision', axis=1)
# y = data['decision']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#
# model = LogisticRegression(solver='liblinear')
# display_results(model, X_train, X_test, y_train, y_test)
