import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_data(file_path):
    """Loading the heart disease dataset."""
    return pd.read_csv(file_path)


def prepare_data(df):
    """
    Preparing features and binary target.

    Target:
        0 = No heart disease
        1 = Heart disease
    """

    # Removing identifier and dataset-source columns
    df = df.drop(columns=["id", "dataset"])

    # Converting original target (0-4) into binary target
    df["num"] = (df["num"] > 0).astype(int)

    # Separating features and target
    X = df.drop(columns=["num"])
    y = df["num"]

    return X, y


def create_preprocessor(X):
    """Create a leakage-safe preprocessing pipeline."""

    numerical_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "str"]
    ).columns.tolist()

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ]
    )

    return preprocessor