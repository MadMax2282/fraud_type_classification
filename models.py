import os
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def get_classifiers():
    models = {
        "Logistic Regression": LogisticRegression(max_iter=500),
        "Ridge Classifier": RidgeClassifier(),
        "LDA": LinearDiscriminantAnalysis(),
        "Gaussian NB": GaussianNB(),
        "KNN": KNeighborsClassifier(n_neighbors=15, n_jobs=-1),
        "Decision Tree": DecisionTreeClassifier(max_depth=12, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
        "Extra Trees": ExtraTreesClassifier(n_estimators=200, n_jobs=-1, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=80, random_state=42),
        "AdaBoost": AdaBoostClassifier(n_estimators=120, random_state=42),
        "SVM (RBF)": SVC(kernel="rbf", C=1.0, random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200, random_state=42),
    }
    return models


import joblib
import pandas as pd

def train_and_evaluate(x_train, x_test, y_train, y_test, class_names):
    print("=" * 60)
    print("3. Навчання та дослідження методів класифікації")
    print("=" * 60)

    models_dir = "models"
    reports_dir = "reports"
    cm_dir = os.path.join("visualizations", "results")
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(cm_dir, exist_ok=True)

    models = get_classifiers()
    results = []
    predictions = {}

    for name, model in models.items():
        print("-" * 60)
        print(f"Метод: {name}")
        print("Навчання моделі...")

        start = time.time()
        model.fit(x_train, y_train)
        train_time = time.time() - start

        # Save model
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
        model_path = os.path.join(models_dir, f"{safe_name}.pkl")
        joblib.dump(model, model_path)

        y_pred = model.predict(x_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"Час навчання: {train_time:.3f} с")
        print(f"Точність на тестовому наборі: {acc:.4f}")
        
        # Classification report
        report_dict = classification_report(y_test, y_pred, target_names=class_names, zero_division=0, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        report_path = os.path.join(reports_dir, f"{safe_name}_report.csv")
        report_df.to_csv(report_path)
        print(f"Звіт збережено у: {report_path}")

        save_confusion_matrix(y_test, y_pred, class_names, name, cm_dir, safe_name)

        results.append({"model": name, "accuracy": acc, "train_time": train_time})
        predictions[name] = y_pred

    return results, predictions


def save_confusion_matrix(y_test, y_pred, class_names, name, cm_dir, safe_name):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(f"Матриця невідповідностей: {name}")
    plt.xlabel("Прогнозований клас")
    plt.ylabel("Справжній клас")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    path = os.path.join(cm_dir, f"cm_{safe_name}.png")
    plt.savefig(path, dpi=120)
    plt.close()
