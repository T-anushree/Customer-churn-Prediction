import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Churn Dashboard", layout="wide")

# -----------------------------
# CUSTOM CSS (UI Styling)
# -----------------------------
st.markdown("""
<style>
.big-title {
    font-size:32px !important;
    font-weight: bold;
    color: #4CAF50;
}
.card {
    padding: 15px;
    border-radius: 10px;
    background-color: #f0f2f6;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-title">📊 Customer Churn Prediction Dashboard</p>', unsafe_allow_html=True)

# -----------------------------
# LOAD MODEL
# -----------------------------
if not os.path.exists("model.pkl"):
    st.error("❌ model.pkl not found. Run train_model.py first.")
    st.stop()

if os.path.getsize("model.pkl") == 0:
    st.error("❌ model.pkl is empty. Re-train the model.")
    st.stop()

with open("model.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
encoders = data["encoders"]
accuracy = data.get("accuracy")
cm = data.get("confusion_matrix")
fpr = data.get("fpr")
tpr = data.get("tpr")
roc_auc = data.get("roc_auc")

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("⚙️ Controls")
show_data = st.sidebar.checkbox("Show Dataset")

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("dataset.csv")
df.columns = df.columns.str.strip().str.lower()
df['churn'] = df['churn'].replace({'Yes': 1, 'No': 0})

# -----------------------------
# TABS
# -----------------------------
tab1, tab2 = st.tabs(["🔮 Prediction", "📊 Dashboard"])

# =============================
# 🔮 PREDICTION TAB
# =============================
with tab1:
    st.header("🔮 Predict Customer Churn")

    age = st.number_input("Age", 18, 100)
    income = st.number_input("Income")
    spendingscore = st.number_input("Spending Score")
    days = st.number_input("Days Since Last Purchase")

    gender = st.selectbox("Gender", ["Male", "Female"])
    country = st.selectbox("Country", ["India"])

    try:
        gender_encoded = encoders['gender'].transform([gender])[0]
        country_encoded = encoders['country'].transform([country])[0]
    except:
        st.error("⚠️ Encoding mismatch. Check training data.")
        st.stop()

    if st.button("Predict Churn"):
        features = np.array([[age, gender_encoded, country_encoded, income, spendingscore, days]])

        prediction = model.predict(features)
        probability = model.predict_proba(features)[0][1]

        st.subheader("🔍 Prediction Result")

        if prediction[0] == 1:
            st.error(f"⚠️ Customer likely to churn\n\nProbability: {probability:.2f}")
        else:
            st.success(f"✅ Customer likely to stay\n\nConfidence: {1 - probability:.2f}")

    if accuracy:
        st.info(f"🎯 Model Accuracy: {accuracy:.2f}")

# =============================
# 📊 DASHBOARD TAB
# =============================
with tab2:
    st.header("📊 Data Dashboard")

    if show_data:
        st.write(df.head())

    # -----------------------------
    # METRICS CARDS
    # -----------------------------
    col1, col2, col3 = st.columns(3)

    col1.markdown(f'<div class="card">👥 Customers<br><b>{len(df)}</b></div>', unsafe_allow_html=True)
    col2.markdown(f'<div class="card">📉 Churn Rate<br><b>{df["churn"].mean():.2f}</b></div>', unsafe_allow_html=True)
    col3.markdown(f'<div class="card">💰 Avg Income<br><b>{df["income"].mean():.0f}</b></div>', unsafe_allow_html=True)

    # -----------------------------
    # GRAPHS
    # -----------------------------
    st.subheader("Churn Distribution")
    fig1, ax1 = plt.subplots()
    df['churn'].value_counts().plot(kind='bar', ax=ax1)
    st.pyplot(fig1)

    st.subheader("Income vs Churn")
    fig2, ax2 = plt.subplots()
    ax2.scatter(df['income'], df['churn'])
    st.pyplot(fig2)

    st.subheader("Spending Score vs Churn")
    fig3, ax3 = plt.subplots()
    ax3.scatter(df['spendingscore'], df['churn'])
    st.pyplot(fig3)

    # -----------------------------
    # CONFUSION MATRIX
    # -----------------------------
    if cm is not None:
        st.subheader("📉 Confusion Matrix")

        fig_cm, ax_cm = plt.subplots()
        ax_cm.imshow(cm)

        for i in range(len(cm)):
            for j in range(len(cm)):
                ax_cm.text(j, i, cm[i][j], ha="center", va="center")

        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")

        st.pyplot(fig_cm)

    # -----------------------------
    # ROC CURVE
    # -----------------------------
    if fpr is not None:
        st.subheader("📉 ROC Curve")

        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        ax_roc.plot([0, 1], [0, 1], linestyle="--")

        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.legend()

        st.pyplot(fig_roc)

    # -----------------------------
    # FEATURE IMPORTANCE
    # -----------------------------
    st.subheader("🧠 Feature Importance")

    features = ["age", "gender", "country", "income", "spendingscore", "days"]
    importances = model.feature_importances_

    fig_imp, ax_imp = plt.subplots()
    ax_imp.barh(features, importances)

    st.pyplot(fig_imp)