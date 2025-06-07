import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from imblearn.over_sampling import SMOTE

# === Load dataset ===
df = pd.read_csv('bank_fraud.csv')
df = df.sample(frac=0.3, random_state=42)  # Sample for speed

# === Drop unhelpful identifier columns ===
df.drop(columns=['Customer_ID', 'Customer_Name', 'Customer_Contact', 'Customer_Email'], inplace=True)

# === Label encode categorical columns ===
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# === Split features and label ===
X = df.drop('Is_Fraud', axis=1)
y = df['Is_Fraud']

# === Scale features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# === Apply SMOTE to handle imbalance ===
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# === Define 5 fast classifiers ===
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Random Forest": RandomForestClassifier(n_estimators=30, n_jobs=-1, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
    "Naive Bayes": GaussianNB(),
    "XGBoost": XGBClassifier(n_estimators=30, use_label_encoder=False, eval_metric='logloss', verbosity=0)
}

# === Train and evaluate each model ===
for name, model in models.items():
    print(f"\n🚀 Training: {name}")
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"✅ Accuracy:  {acc:.4f}")
    print(f"✅ Precision: {prec:.4f}")
    print(f"✅ Recall:    {rec:.4f}")
    print(f"✅ F1 Score:  {f1:.4f}")
    print("📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("📄 Classification Report:\n", classification_report(y_test, y_pred))

# === Train final model: Stacking Classifier ===
stacking = StackingClassifier(estimators=[
    ('lr', LogisticRegression(max_iter=500)),
    ('dt', DecisionTreeClassifier(max_depth=10)),
    ('nb', GaussianNB())
], final_estimator=LogisticRegression())

print("\n🚀 Training: Stacking Classifier")
stacking.fit(X_train_res, y_train_res)
y_pred_stack = stacking.predict(X_test)

acc = accuracy_score(y_test, y_pred_stack)
prec = precision_score(y_test, y_pred_stack, zero_division=0)
rec = recall_score(y_test, y_pred_stack, zero_division=0)
f1 = f1_score(y_test, y_pred_stack, zero_division=0)

print(f"\n✅ Stacking Classifier Evaluation:")
print(f"✅ Accuracy:  {acc:.4f}")
print(f"✅ Precision: {prec:.4f}")
print(f"✅ Recall:    {rec:.4f}")
print(f"✅ F1 Score:  {f1:.4f}")
print("📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred_stack))
print("📄 Classification Report:\n", classification_report(y_test, y_pred_stack))

# === Save model, scaler, and encoders ===
print("\n💾 Saving model and preprocessing objects...")

joblib.dump(stacking, 'stacking_classifier_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

for col, le in label_encoders.items():
    joblib.dump(le, f'enc_{col}.pkl')

print("✅ All objects saved successfully.")
