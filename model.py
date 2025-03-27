import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv('quikr_car.csv')

# Preprocessing (Ensure columns match)
df = df.dropna()  # Remove null values
df['year'] = df['year'].astype(int)

# Fix: Only apply .str.replace() if "kms_driven" is a string column
if df['kms_driven'].dtype == 'O':  
    df['kms_driven'] = df['kms_driven'].str.replace(',', '').astype(float)

df['company'] = df['company'].astype('category').cat.codes
df['fuel_type'] = df['fuel_type'].astype('category').cat.codes

# Features and Target
X = df[['year', 'kms_driven', 'company', 'fuel_type']]
y = df['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
with open('car_price_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("✅ Model trained and saved successfully!")
