from src.preprocessing import (
    load_data,
    prepare_data,
    create_preprocessor,
)


def test_load_data():
    df = load_data("data/heart.csv")

    assert df.shape == (920, 16)


def test_prepare_data():
    df = load_data("data/heart.csv")

    X, y = prepare_data(df)

    assert X.shape == (920, 13)
    assert len(y) == 920


def test_target_is_binary():
    df = load_data("data/heart.csv")

    X, y = prepare_data(df)

    assert set(y.unique()) == {0, 1}


def test_removed_columns():
    df = load_data("data/heart.csv")

    X, y = prepare_data(df)

    assert "id" not in X.columns
    assert "dataset" not in X.columns
    assert "num" not in X.columns


def test_preprocessor_creation():
    df = load_data("data/heart.csv")

    X, y = prepare_data(df)

    preprocessor = create_preprocessor(X)

    assert preprocessor is not None


def test_preprocessor_transformation():
    df = load_data("data/heart.csv")

    X, y = prepare_data(df)

    preprocessor = create_preprocessor(X)

    X_transformed = preprocessor.fit_transform(X)

    assert X_transformed.shape[0] == 920
    assert X_transformed.shape[1] > 0