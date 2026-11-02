
import pandas as pd
import numpy as np

# Load the original dataset
df = pd.read_excel("data/customer_churn.xlsx")

print("Original dataset shape:", df.shape)

# Remove customer ID because it is not useful for prediction
df = df.drop(columns=["Customer_ID"])

# Feature 1: Create tenure categories
df["Tenure_Category"] = pd.cut(
    df["Tenure_Months"],
    bins=[-np.inf, 12, 24, 48, np.inf],
    labels=["0-12 Months", "13-24 Months",
            "25-48 Months", "Above 48 Months"]
)

# Feature 2: Calculate average monthly spending
df["Avg_Monthly_Spending_INR"] = np.where(
    df["Tenure_Months"] > 0,
    df["Total_Charges_INR"] / df["Tenure_Months"],
    np.nan
)

# Feature 3: Create service usage indicators
def service_indicator(series):
    values = series.astype("string").str.strip().str.lower()
    return values.map({
        "yes": 1,
        "no": 0,
        "no internet service": 0,
        "no phone service": 0
    })

df["Has_Tech_Support"] = service_indicator(df["Tech_Support"])
df["Has_Online_Security"] = service_indicator(df["Online_Security"])

internet = df["Internet_Service"].astype("string").str.strip().str.lower()
df["Has_Internet"] = internet.map({
    "dsl": 1,
    "fiber optic": 1,
    "yes": 1,
    "no": 0
})

# Display processed dataset information
print("\nProcessed dataset shape:", df.shape)

print("\nNew feature columns:")
print([
    "Tenure_Category",
    "Avg_Monthly_Spending_INR",
    "Has_Tech_Support",
    "Has_Online_Security",
    "Has_Internet"
])

print("\nMissing values after feature engineering:")
print(df.isnull().sum())

print("\nFirst five processed records:")
print(df.head())

# Save processed dataset
df.to_csv("data/processed_churn.csv", index=False)

print("\nProcessed dataset saved successfully!")
