import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_predict
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from src.preprocessing import (
    load_data,
    prepare_data,
    create_preprocessor
)


# --------------------------------------------------
# Helper function
# --------------------------------------------------

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


# --------------------------------------------------
# Calculating the threshold metrics
# --------------------------------------------------

def calculate_threshold_metrics(
    y_true,
    probabilities,
    threshold
):
    """
    Calculate classification metrics for a
    particular probability threshold.
    """

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions
    ).ravel()

    return {
        "Threshold": round(threshold, 2),
        "Accuracy": accuracy_score(
            y_true,
            predictions
        ),
        "Recall": recall_score(
            y_true,
            predictions,
            zero_division=0
        ),
        "Precision": precision_score(
            y_true,
            predictions,
            zero_division=0
        ),
        "Specificity": calculate_specificity(
            y_true,
            predictions
        ),
        "F1": f1_score(
            y_true,
            predictions,
            zero_division=0
        ),
        "False Negatives": fn,
        "False Positives": fp,
        "True Negatives": tn,
        "True Positives": tp
    }


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    # --------------------------------------------------
    # 1. Create reports directory
    # --------------------------------------------------

    os.makedirs("reports", exist_ok=True)

    # --------------------------------------------------
    # 2. Load and prepare data
    # --------------------------------------------------

    df = load_data("data/heart.csv")

    X, y = prepare_data(df)

    # --------------------------------------------------
    # 3. Train/test split
    #
    # The test set is kept completely untouched
    # during threshold selection.
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42
    )

    print("\n" + "=" * 80)
    print("DATA SPLIT")
    print("=" * 80)

    print(f"Total samples:    {len(X)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}")

    # --------------------------------------------------
    # 4. Create preprocessing
    # --------------------------------------------------

    preprocessor = create_preprocessor(
        X_train
    )

    # --------------------------------------------------
    # 5. Create Random Forest model
    # --------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model)
        ]
    )

    # --------------------------------------------------
    # 6. Cross-validation ONLY on training data
    # --------------------------------------------------

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    print("\n" + "=" * 80)
    print("CROSS-VALIDATED THRESHOLD ANALYSIS")
    print("=" * 80)

    print(
        "\nGenerating out-of-fold probabilities "
        "using training data only..."
    )

    probabilities = cross_val_predict(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1
    )[:, 1]

    print(
        "Cross-validation completed."
    )

    # --------------------------------------------------
    # 7. Test thresholds
    # --------------------------------------------------

    thresholds = np.arange(
        0.20,
        0.701,
        0.01
    )

    results = []

    for threshold in thresholds:

        metrics = calculate_threshold_metrics(
            y_train,
            probabilities,
            threshold
        )

        results.append(metrics)

    results_df = pd.DataFrame(results)

    # --------------------------------------------------
    # 8. Display threshold results
    # --------------------------------------------------

    display_columns = [
        "Threshold",
        "Recall",
        "Precision",
        "Specificity",
        "F1",
        "False Negatives",
        "False Positives"
    ]

    print(
        results_df[
            display_columns
        ].round(4).to_string(index=False)
    )

    # --------------------------------------------------
    # 9. Select threshold
    #
    # Requirement:
    # Recall >= 95%
    #
    # Among eligible thresholds, choose the highest
    # threshold. This reduces false positives while
    # still satisfying the recall requirement.
    # --------------------------------------------------

    eligible = results_df[
        results_df["Recall"] >= 0.95
    ]

    if eligible.empty:

        print(
            "\nNo threshold achieved the target "
            "recall of 95%."
        )

        # Fallback:
        # choose threshold with maximum recall
        selected = results_df.loc[
            results_df["Recall"].idxmax()
        ]

        selection_reason = (
            "No threshold achieved 95% recall; "
            "selected threshold with maximum recall."
        )

    else:

        selected = eligible.iloc[-1]

        selection_reason = (
            "Highest threshold achieving at least "
            "95% recall."
        )

    selected_threshold = float(
        selected["Threshold"]
    )

    # --------------------------------------------------
    # 10. Display selected threshold
    # --------------------------------------------------

    print("\n" + "=" * 80)
    print("SELECTED THRESHOLD")
    print("=" * 80)

    print(
        f"Threshold:     {selected_threshold:.2f}"
    )

    print(
        f"Recall:        {selected['Recall']:.4f}"
    )

    print(
        f"Precision:     {selected['Precision']:.4f}"
    )

    print(
        f"Specificity:   {selected['Specificity']:.4f}"
    )

    print(
        f"F1:            {selected['F1']:.4f}"
    )

    print(
        f"False Negatives: "
        f"{int(selected['False Negatives'])}"
    )

    print(
        f"False Positives: "
        f"{int(selected['False Positives'])}"
    )

    print(
        f"\nSelection rule: {selection_reason}"
    )

    # --------------------------------------------------
    # 11. Save threshold analysis
    # --------------------------------------------------

    threshold_file = (
        "reports/threshold_analysis.csv"
    )

    results_df.to_csv(
        threshold_file,
        index=False
    )

    print(
        f"\nThreshold results saved to: "
        f"{threshold_file}"
    )

    # --------------------------------------------------
    # 12. Create threshold performance plot
    # --------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.plot(
        results_df["Threshold"],
        results_df["Recall"],
        marker="o",
        label="Recall"
    )

    plt.plot(
        results_df["Threshold"],
        results_df["Precision"],
        marker="o",
        label="Precision"
    )

    plt.plot(
        results_df["Threshold"],
        results_df["Specificity"],
        marker="o",
        label="Specificity"
    )

    plt.axvline(
        selected_threshold,
        linestyle="--",
        label=(
            f"Selected threshold = "
            f"{selected_threshold:.2f}"
        )
    )

    plt.xlabel("Classification Threshold")
    plt.ylabel("Score")

    plt.title(
        "Random Forest Threshold Analysis"
    )

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_file = (
        "reports/threshold_analysis.png"
    )

    plt.savefig(
        plot_file,
        dpi=300
    )

    plt.close()

    print(
        f"Threshold plot saved to: "
        f"{plot_file}"
    )

    # --------------------------------------------------
    # 13. Train final Random Forest
    #     using training data only
    # --------------------------------------------------

    print("\n" + "=" * 80)
    print("FINAL TEST-SET EVALUATION")
    print("=" * 80)

    pipeline.fit(
        X_train,
        y_train
    )

    # Probability predictions on untouched test data
    test_probabilities = pipeline.predict_proba(
        X_test
    )[:, 1]

    # --------------------------------------------------
    # 14. Evaluate default threshold = 0.50
    # --------------------------------------------------

    default_metrics = calculate_threshold_metrics(
        y_test,
        test_probabilities,
        0.50
    )

    # --------------------------------------------------
    # 15. Evaluate selected threshold
    # --------------------------------------------------

    selected_metrics = calculate_threshold_metrics(
        y_test,
        test_probabilities,
        selected_threshold
    )

    # --------------------------------------------------
    # 16. Display comparison
    # --------------------------------------------------

    comparison = pd.DataFrame(
        [
            {
                "Threshold": 0.50,
                "Accuracy": default_metrics["Accuracy"],
                "Recall": default_metrics["Recall"],
                "Precision": default_metrics["Precision"],
                "Specificity": default_metrics["Specificity"],
                "F1": default_metrics["F1"],
                "False Negatives": default_metrics[
                    "False Negatives"
                ],
                "False Positives": default_metrics[
                    "False Positives"
                ]
            },
            {
                "Threshold": selected_threshold,
                "Accuracy": selected_metrics["Accuracy"],
                "Recall": selected_metrics["Recall"],
                "Precision": selected_metrics["Precision"],
                "Specificity": selected_metrics["Specificity"],
                "F1": selected_metrics["F1"],
                "False Negatives": selected_metrics[
                    "False Negatives"
                ],
                "False Positives": selected_metrics[
                    "False Positives"
                ]
            }
        ]
    )

    print(
        comparison.round(4).to_string(
            index=False
        )
    )

    # --------------------------------------------------
    # 17. Display selected threshold confusion matrix
    # --------------------------------------------------

    selected_predictions = (
        test_probabilities >= selected_threshold
    ).astype(int)

    selected_cm = confusion_matrix(
        y_test,
        selected_predictions
    )

    print(
        "\nConfusion Matrix "
        f"(threshold = {selected_threshold:.2f}):"
    )

    print(selected_cm)

    tn, fp, fn, tp = selected_cm.ravel()

    print("\nClinical interpretation:")
    print(f"True Negatives:  {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives:  {tp}")

    # --------------------------------------------------
    # 18. ROC-AUC
    # --------------------------------------------------

    roc_auc = roc_auc_score(
        y_test,
        test_probabilities
    )

    print(
        f"\nROC-AUC: {roc_auc:.4f}"
    )

    # --------------------------------------------------
    # 19. Save final comparison
    # --------------------------------------------------

    comparison_file = (
        "reports/threshold_comparison.csv"
    )

    comparison.to_csv(
        comparison_file,
        index=False
    )

    print(
        f"\nFinal comparison saved to: "
        f"{comparison_file}"
    )

    # --------------------------------------------------
    # 20. Final summary
    # --------------------------------------------------

    print("\n" + "=" * 80)
    print("FINAL THRESHOLD DECISION")
    print("=" * 80)

    print(
        f"Selected threshold: "
        f"{selected_threshold:.2f}"
    )

    print(
        "Reason: prioritize high recall because "
        "false negatives are more costly."
    )

    print(
        f"Test-set recall: "
        f"{selected_metrics['Recall']:.4f}"
    )

    print(
        f"Test-set precision: "
        f"{selected_metrics['Precision']:.4f}"
    )

    print(
        f"Test-set specificity: "
        f"{selected_metrics['Specificity']:.4f}"
    )

    print(
        f"Test-set false negatives: "
        f"{int(selected_metrics['False Negatives'])}"
    )

    print(
        f"Test-set false positives: "
        f"{int(selected_metrics['False Positives'])}"
    )


if __name__ == "__main__":
    main()