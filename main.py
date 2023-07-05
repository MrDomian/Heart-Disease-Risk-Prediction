from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score, recall_score, confusion_matrix


def display_results(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    sensitivity = recall_score(y_test, y_pred)
    specificity = roc_auc

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    g_mean = (tp / (tp + fn)) * (tn / (tn + fp))

    print("Accuracy:", round(accuracy, 2))
    print("Precision:", round(precision, 2))
    print("F1 Score:", round(f1, 2))
    print("ROC AUC Score:", round(roc_auc, 2))
    print("Sensitivity:", round(sensitivity, 2))
    print("Specificity:", round(specificity, 2))
    print("True Negative:", tn)
    print("False Positive:", fp)
    print("False Negative:", fn)
    print("True Positive:", tp)
    print("G-Mean:", round(g_mean, 2))
