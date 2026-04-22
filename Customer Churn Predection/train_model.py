import pandas as pd
import numpy as np
from datetime import datetime
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("dataset.csv")

# -----------------------------
# CLEAN COLUMN NAMES
# -----------------------------
df.columns = df.columns.str.strip().str.lower()

# -----------------------------
# HANDLE MISSING VALUES
# -----------------------------
df = df.ffill()

# -----------------------------
# FIX GENDER
# -----------------------------
df['gender'] = df['gender'].replace({
    'M': 'Male',
    'F': 'Female'
})
df['gender'] = df['gender'].fillna('Male')

# -----------------------------
# FIX COUNTRY
# -----------------------------
df['country'] = df['country'].replace({
    'IND': 'India',
    'IN': 'India'
})
df['country'] = df['country'].fillna('India')

# -----------------------------
# FIX CHURN (TARGET)
# -----------------------------
df['churn'] = df['churn'].replace({
    'Yes': 1,
    'No': 0
})

df['churn'] = pd.to_numeric(df['churn'], errors='coerce')
df = df.dropna(subset=['churn'])
df['churn'] = df['churn'].astype(int)

# -----------------------------
# FIX DATE FORMAT
# -----------------------------
df['lastpurchasedate'] = pd.to_datetime(
    df['lastpurchasedate'], errors='coerce'
)

df = df.dropna(subset=['lastpurchasedate'])

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------
df['days_since_last_purchase'] = (
    datetime.now() - df['lastpurchasedate']
).dt.days

# -----------------------------
# SELECT ONLY REQUIRED FEATURES ✅
# -----------------------------
df = df[[
    'age',
    'gender',
    'country',
    'income',
    'spendingscore',
    'days_since_last_purchase',
    'churn'
]]

# -----------------------------
# ENCODING
# -----------------------------
encoders = {}

for col in ['gender', 'country']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# -----------------------------
# SPLIT
# -----------------------------
X = df.drop('churn', axis=1)
y = df['churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Features used:", X.shape[1])  # should be 6

# -----------------------------
# MODEL
# -----------------------------
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# EVALUATION
# -----------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

print(f"✅ Model Accuracy: {accuracy:.2f}")
print("✅ Confusion Matrix:\n", cm)
print(f"✅ ROC AUC: {roc_auc:.2f}")

# -----------------------------
# SAVE EVERYTHING (MATCHES app.py)
# -----------------------------
with open("model.pkl", "wb") as f:
    pickle.dump({
        "model": model,
        "encoders": encoders,
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": roc_auc
    }, f)

print("🚀 model.pkl saved successfully!")