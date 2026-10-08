import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocessing import (
    load_data,
    prepare_data,
    create_preprocessor
)

from src.models import get_models


def main():

    # --------------------------------------------------
    # 1. Loading and preparing data
    # --------------------------------------------------

    df = load_data("data/heart.csv")

    X, y = prepare_data(df)

    # --------------------------------------------------
    # 2. Training/testing split
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42
    )

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    print("\nTraining target distribution:")
    print(y_train.value_counts())

    print("\nTesting target distribution:")
    print(y_test.value_counts())

    # --------------------------------------------------
    # 3. Creating preprocessing
    # --------------------------------------------------

    preprocessor = create_preprocessor(X_train)

    # --------------------------------------------------
    # 4. Getting models
    # --------------------------------------------------

    models = get_models()

    # --------------------------------------------------
    # 5. Training each model
    # --------------------------------------------------

    trained_models = {}

    for name, model in models.items():

        print(f"\nTraining {name}...")

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", model)
            ]
        )

        pipeline.fit(X_train, y_train)

        trained_models[name] = pipeline

        print(f"{name} trained successfully.")

    print("\nAll models trained successfully.")

    return trained_models, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    main()