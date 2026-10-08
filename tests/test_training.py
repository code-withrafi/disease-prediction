from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocessing import (
    load_data,
    prepare_data,
    create_preprocessor,
)

from src.models import get_models


def test_all_models_can_train_and_predict():
    # Load and prepare data
    df = load_data("data/heart.csv")
    X, y = prepare_data(df)

    # Small stratified split for testing
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )

    models = get_models()

    for name, model in models.items():

        preprocessor = create_preprocessor(X_train)

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", model),
            ]
        )

        # Train
        pipeline.fit(X_train, y_train)

        # Predict
        predictions = pipeline.predict(X_test)

        # Predict probabilities
        probabilities = pipeline.predict_proba(X_test)[:, 1]

        # Basic validation
        assert len(predictions) == len(y_test)
        assert len(probabilities) == len(y_test)

        assert set(predictions).issubset({0, 1})

        assert ((probabilities >= 0) & (probabilities <= 1)).all()