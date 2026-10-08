from src.models import get_models


def test_get_models():
    models = get_models()

    assert len(models) == 3


def test_model_names():
    models = get_models()

    expected_models = {
        "Logistic Regression",
        "Random Forest",
        "XGBoost",
    }

    assert set(models.keys()) == expected_models


def test_models_have_predict_proba():
    models = get_models()

    for name, model in models.items():
        assert hasattr(model, "predict_proba"), (
            f"{name} does not support probability prediction"
        )


def test_models_are_created_successfully():
    models = get_models()

    for name, model in models.items():
        assert model is not None