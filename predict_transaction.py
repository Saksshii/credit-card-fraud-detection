import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

# === Load the saved model and scaler ===
model = joblib.load('stacking_classifier_model.pkl')
scaler = joblib.load('scaler.pkl')

# === Define all feature columns in the right order ===
feature_names = [
    'Gender', 'Age', 'State', 'City', 'Bank_Branch', 'Account_Type',
    'Transaction_Amount', 'Transaction_Type', 'Merchant_Category',
    'Account_Balance', 'Transaction_Device', 'Transaction_Location',
    'Device_Type', 'Transaction_Currency', 'Transaction_Description'
]

# === Load saved encoders for categorical columns ===
categorical_cols = [
    'Gender', 'State', 'City', 'Bank_Branch', 'Account_Type',
    'Transaction_Type', 'Merchant_Category', 'Transaction_Device',
    'Transaction_Location', 'Device_Type', 'Transaction_Currency', 'Transaction_Description'
]

label_encoders = {}
for col in categorical_cols:
    label_encoders[col] = joblib.load(f'enc_{col}.pkl')

# === Get user input ===
print("\n📥 Enter transaction details:")
user_input = {}
for feature in feature_names:
    val = input(f"{feature}: ").strip()
    user_input[feature] = val



# === Create input DataFrame ===
input_df = pd.DataFrame([user_input])

# === Encode categorical values ===
for col in categorical_cols:
    if col in input_df.columns:
        try:
            input_df[col] = label_encoders[col].transform(input_df[col])
        except:
            print(f"⚠️  Unknown value '{input_df[col][0]}' for column '{col}'. Please enter a known value.")
            exit(1)

# === Convert numerics ===
numeric_cols = ['Age', 'Transaction_Amount', 'Account_Balance']
for col in numeric_cols:
    input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

# === Scale input ===
input_scaled = scaler.transform(input_df)

# === Predict ===
prediction = model.predict(input_scaled)[0]

# === Output Result ===
print("\n🔍 Prediction Result:")
if prediction == 1:
    print("⚠️ Fraudulent Transaction Detected!")
else:
    print("✅ Legitimate Transaction.")
