import os

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from src.train import main


def calculate_specificity(y_true, y_pred):
    """
    Calculate specificity:
        TN / (TN + FP)
    """

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred
    ).ravel()

    return tn / (tn + fp)


def evaluate_models():
    """
    Train and evaluate all models.

    Results are printed and saved to:
        reports/model_comparison.csv
    """

    trained_models, X_train, X_test, y_train, y_test = main()

    results = []

    # Create reports directory if it does not exist
    os.makedirs("reports", exist_ok=True)

    for name, model in trained_models.items():

        # Default classification threshold = 0.50
        y_pred = model.predict(X_test)

        # Probability of disease
        y_probability = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        specificity = calculate_specificity(
            y_test,
            y_pred
        )

        roc_auc = roc_auc_score(
            y_test,
            y_probability
        )

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "Specificity": specificity,
            "F1": f1,
            "ROC-AUC": roc_auc
        })

        print("\n" + "=" * 60)
        print(name)
        print("=" * 60)

        print(f"Accuracy:    {accuracy:.4f}")
        print(f"Precision:   {precision:.4f}")
        print(f"Recall:      {recall:.4f}")
        print(f"Specificity:  {specificity:.4f}")
        print(f"F1 Score:    {f1:.4f}")
        print(f"ROC-AUC:     {roc_auc:.4f}")

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Save results
    output_path = "reports/model_comparison.csv"

    results_df.to_csv(
        output_path,
        index=False
    )

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print(
        results_df.round(4).to_string(index=False)
    )

    print(f"\nResults saved to: {output_path}")

    return results_df


if __name__ == "__main__":
    evaluate_models()