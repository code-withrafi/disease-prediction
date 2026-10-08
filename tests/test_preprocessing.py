from src.preprocessing import load_data, prepare_data, create_preprocessor


# Load data
df = load_data("data/heart.csv")

print("Original shape:", df.shape)

# Prepare features and target
X, y = prepare_data(df)

print("Feature shape:", X.shape)
print("Target shape:", y.shape)

print("\nTarget distribution:")
print(y.value_counts())

print("\nFeatures:")
print(X.columns.tolist())

# Create preprocessing pipeline
preprocessor = create_preprocessor(X)

print("\nPreprocessor created successfully.")