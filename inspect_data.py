import pandas as pd

# Load dataset
df = pd.read_csv("data/heart.csv")

print("=" * 60)
print("DATASET SHAPE")
print("=" * 60)
print(df.shape)

print("\n" + "=" * 60)
print("COLUMN NAMES")
print("=" * 60)
print(df.columns.tolist())

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)
print(df.isnull().sum())

print("\n" + "=" * 60)
print("UNIQUE VALUES")
print("=" * 60)

categorical_columns = [
    "sex",
    "dataset",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "thal"
]

for column in categorical_columns:
    print(f"\n{column}:")
    print(df[column].value_counts(dropna=False))

print("\n" + "=" * 60)
print("TARGET DISTRIBUTION")
print("=" * 60)

print(df["num"].value_counts().sort_index())

print("\n" + "=" * 60)
print("TARGET PERCENTAGES")
print("=" * 60)

print(
    (df["num"].value_counts(normalize=True).sort_index() * 100)
    .round(2)
)