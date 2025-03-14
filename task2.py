import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

# Load dataset
df = pd.read_csv(r"C:\Users\vaish\Downloads\amazon.csv 2\amazon.csv")  # Change filename if necessary

# Print column names
print("Columns in dataset:", df.columns)

# Convert numeric columns properly
numeric_cols = ["discounted_price", "actual_price", "discount_percentage", "rating", "rating_count"]

# Force conversion of numeric columns, replacing errors with NaN
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Check if conversion worked
print("Data types after conversion:\n", df.dtypes)

# Identify problematic values
for col in numeric_cols:
    if col in df.columns:
        print(f"Unique values in {col}:\n", df[col].unique())

# Drop completely empty columns
df.dropna(axis=1, how="all", inplace=True)

# Drop unnecessary text columns
drop_cols = ["product_id", "product_name", "user_name", "review_content", 
             "review_title", "product_link", "img_link", "review_id", "about_product", "user_id"]
df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True, errors="ignore")

# Encode categorical columns
if "category" in df.columns:
    le = LabelEncoder()
    df["category"] = le.fit_transform(df["category"].astype(str))

# Handle missing values only for existing numeric columns
existing_numeric_cols = [col for col in numeric_cols if col in df.columns]
if existing_numeric_cols:
    imputer = SimpleImputer(strategy="mean")
    df[existing_numeric_cols] = imputer.fit_transform(df[existing_numeric_cols])

# Check for missing values
print("Missing values in dataset:\n", df.isnull().sum())

# Define features (X) and target (y)
if "rating_count" in df.columns:
    X = df.drop(columns=["rating_count"])  
    y = df["rating_count"]  
else:
    raise ValueError("Target column 'rating_count' is missing from the dataset.")

# Split data into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and Evaluate
y_pred = model.predict(X_test)
print("Predictions:", y_pred[:5])  # Show first 5 predictions
