
import pandas as pd
from sklearn.model_selection import train_test_split

# Load processed dataset
df = pd.read_csv("data/processed_churn.csv")

# Separate features and target
X = df.drop(columns=["Churn"])
y = df["Churn"]

# Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Display dataset sizes
print("Training and testing split completed!")

print("Total records:", len(df))
print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())

# Save split datasets
X_train.to_csv("data/X_train.csv", index=False)
X_test.to_csv("data/X_test.csv", index=False)
y_train.to_csv("data/y_train.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)

print("\nTraining and testing datasets saved successfully!")
