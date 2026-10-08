import os

import pandas as pd
import matplotlib.pyplot as plt


def create_model_comparison_plot():
    """
    Create a bar chart comparing the three classification models.
    """

    input_path = "reports/model_comparison.csv"
    output_path = "reports/model_comparison.png"

    # Load evaluation results
    results = pd.read_csv(input_path)

    # Metrics to visualize
    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "Specificity",
        "F1",
        "ROC-AUC",
    ]

    # Create reports directory if necessary
    os.makedirs("reports", exist_ok=True)

    # Create one figure
    ax = results.set_index("Model")[metrics].plot(
        kind="bar",
        figsize=(12, 6)
    )

    ax.set_title("Model Performance Comparison")
    ax.set_ylabel("Score")
    ax.set_xlabel("Model")
    ax.set_ylim(0, 1)

    plt.xticks(rotation=0)
    plt.legend(
        title="Metric",
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    plt.tight_layout()

    # Save figure
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Plot saved to: {output_path}")


if __name__ == "__main__":
    create_model_comparison_plot()