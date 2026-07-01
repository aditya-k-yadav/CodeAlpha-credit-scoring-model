"""
CodeAlpha Machine Learning Internship
Task 1: Credit Scoring Model
--------------------------------------
Objective: Predict an individual's creditworthiness (good/bad credit risk)
using financial history data such as income, debts, payment history, etc.

Algorithms used: Logistic Regression, Decision Tree, Random Forest
Evaluation metrics: Precision, Recall, F1-Score, ROC-AUC

Note: This script generates a realistic synthetic financial dataset so it
runs end-to-end with no internet connection or external downloads. You can
swap in a real dataset (e.g. the UCI "German Credit Data" or "Give Me Some
Credit" Kaggle dataset) by replacing the `create_dataset()` function with a
`pd.read_csv("your_data.csv")` call - the rest of the pipeline stays the same.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# ---------------------------------------------------------------------
# 1. Create / Load Dataset
# ---------------------------------------------------------------------
def create_dataset(n_samples=2000):
    """Generates a synthetic but realistic credit scoring dataset."""
    age = np.random.randint(21, 65, n_samples)
    annual_income = np.random.normal(55000, 20000, n_samples).clip(15000, 200000)
    existing_debt = np.random.normal(15000, 10000, n_samples).clip(0, 100000)
    credit_history_years = np.random.randint(0, 30, n_samples)
    num_late_payments = np.random.poisson(1.2, n_samples)
    loan_amount = np.random.normal(20000, 12000, n_samples).clip(1000, 80000)
    employment_years = np.random.randint(0, 35, n_samples)
    num_credit_lines = np.random.randint(1, 15, n_samples)

    debt_to_income = existing_debt / annual_income

    # Underlying "true" risk score that drives creditworthiness
    risk_score = (
        -0.00003 * annual_income
        + 0.00004 * existing_debt
        + 3.0 * debt_to_income
        - 0.05 * credit_history_years
        + 0.4 * num_late_payments
        - 0.03 * employment_years
        + 0.00002 * loan_amount
        - 0.02 * num_credit_lines
        + np.random.normal(0, 0.5, n_samples)  # noise
    )

    # Convert risk score into a binary label: 1 = Good credit, 0 = Bad credit
    threshold = np.percentile(risk_score, 65)
    creditworthy = (risk_score < threshold).astype(int)

    df = pd.DataFrame({
        "age": age,
        "annual_income": annual_income.round(2),
        "existing_debt": existing_debt.round(2),
        "credit_history_years": credit_history_years,
        "num_late_payments": num_late_payments,
        "loan_amount": loan_amount.round(2),
        "employment_years": employment_years,
        "num_credit_lines": num_credit_lines,
        "debt_to_income_ratio": debt_to_income.round(3),
        "creditworthy": creditworthy,  # target: 1 = good, 0 = bad
    })
    return df


# ---------------------------------------------------------------------
# 2. Main pipeline
# ---------------------------------------------------------------------
def main():
    print("=" * 60)
    print("TASK 1: CREDIT SCORING MODEL")
    print("=" * 60)

    df = create_dataset()
    df.to_csv("credit_data.csv", index=False)
    print(f"\nDataset created with {len(df)} rows -> saved as credit_data.csv")
    print(df.head())
    print("\nClass balance (1 = creditworthy, 0 = not creditworthy):")
    print(df["creditworthy"].value_counts())

    # Features / target split
    X = df.drop("creditworthy", axis=1)
    y = df["creditworthy"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Scale features (helps Logistic Regression converge well)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ------------------------------------------------------------
    # Train multiple models
    # ------------------------------------------------------------
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=RANDOM_STATE),
    }

    results = []
    fitted_models = {}

    for name, model in models.items():
        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

        fitted_models[name] = model

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        results.append({
            "Model": name, "Accuracy": acc, "Precision": prec,
            "Recall": rec, "F1-Score": f1, "ROC-AUC": roc_auc
        })

        print(f"\n--- {name} ---")
        print(classification_report(y_test, y_pred, target_names=["Bad Credit", "Good Credit"]))
        print(f"ROC-AUC: {roc_auc:.4f}")

    results_df = pd.DataFrame(results).set_index("Model").round(4)
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(results_df)
    results_df.to_csv("model_comparison_results.csv")

    # ------------------------------------------------------------
    # Plot: metric comparison bar chart
    # ------------------------------------------------------------
    results_df.plot(kind="bar", figsize=(10, 6))
    plt.title("Credit Scoring Model Comparison")
    plt.ylabel("Score")
    plt.xticks(rotation=15)
    plt.ylim(0, 1)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=150)
    print("\nSaved chart -> model_comparison.png")

    # ------------------------------------------------------------
    # Plot: confusion matrix for best model (by ROC-AUC)
    # ------------------------------------------------------------
    best_model_name = results_df["ROC-AUC"].idxmax()
    best_model = fitted_models[best_model_name]
    X_eval = X_test_scaled if best_model_name == "Logistic Regression" else X_test
    y_pred_best = best_model.predict(X_eval)

    cm = confusion_matrix(y_test, y_pred_best)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Bad Credit", "Good Credit"],
                yticklabels=["Bad Credit", "Good Credit"])
    plt.title(f"Confusion Matrix - {best_model_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print(f"Best model: {best_model_name} -> saved confusion_matrix.png")

    # ------------------------------------------------------------
    # Feature importance (Random Forest)
    # ------------------------------------------------------------
    rf_model = fitted_models["Random Forest"]
    importance = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values()
    plt.figure(figsize=(8, 6))
    importance.plot(kind="barh", color="teal")
    plt.title("Feature Importance (Random Forest)")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    print("Saved chart -> feature_importance.png")

    print("\nDone! All outputs saved in the current folder.")


if __name__ == "__main__":
    main()
