import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


# ---------------- LOAD DATA ----------------
def load_data(file_path):
    data = pd.read_csv(file_path)

    data.drop(['name', 'subject#'], axis=1, errors='ignore', inplace=True)

    threshold = data['total_UPDRS'].median()
    data['status'] = (data['total_UPDRS'] > threshold).astype(int)

    X = data.drop(['status', 'total_UPDRS', 'motor_UPDRS'], axis=1)
    y = data['status']
    return data, X, y


# ---------------- TRAIN MODELS ----------------
def train_models(file_path, threshold=0.45):
    data, X, y = load_data(file_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = {
        'SVM': SVC(kernel='linear', probability=True),
        'RF': RandomForestClassifier(n_estimators=200, max_depth=8),
        'DT': DecisionTreeClassifier(max_depth=4),
        'KNN': KNeighborsClassifier(n_neighbors=9),
        'GB': GradientBoostingClassifier()
    }

    ensemble = VotingClassifier(
        estimators=[('svm', models['SVM']), ('rf', models['RF']),
                    ('knn', models['KNN']), ('gb', models['GB'])],
        voting='soft'
    )

    models['Ensemble'] = ensemble

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results[name] = accuracy_score(y_test, pred)

    final_model = models['Ensemble']
    y_prob = final_model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "AUC": auc(*roc_curve(y_test, y_prob)[:2]),
        "Confusion Matrix": confusion_matrix(y_test, y_pred)
    }

    return data, X, scaler, final_model, metrics, results


# ---------------- MAIN PROGRAM ----------------
def main():
    print("=== Parkinson's Detection System ===")

    file_path = input("Enter dataset path: ")
    threshold = float(input("Enter threshold (0.1 - 0.9, default 0.45): ") or 0.45)

    data, X, scaler, model, metrics, results = train_models(file_path, threshold)

    print("\n--- MODEL PERFORMANCE ---")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    print("\n--- MODEL COMPARISON ---")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    print("\n--- DATASET INFO ---")
    print(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")

    # -------- PREDICTION --------
    print("\nEnter patient values:")

    values = []
    for col in X.columns:
        val = float(input(f"{col}: "))
        values.append(val)

    values = np.array([values])
    values = scaler.transform(values)

    prob = model.predict_proba(values)[0][1]
    pred = int(prob >= threshold)

    print("\n--- RESULT ---")
    print("Prediction:", "High Risk" if pred else "Low Risk")
    print("Probability:", round(prob, 4))


# RUN
if __name__ == "__main__":
    main()