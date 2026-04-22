# 📊 Customer Churn Prediction System

## 🔍 Problem Statement

Customer churn is a critical business problem where companies lose customers over time. This project aims to predict whether a customer is likely to churn using machine learning, enabling proactive retention strategies.

---

## 📁 Dataset

The dataset contains customer information such as:

* Age
* Gender
* Country
* Income
* Spending Score
* Last Purchase Date

After preprocessing, a new feature is created:

* **Days Since Last Purchase**

Target variable:

* **Churn (0 = No, 1 = Yes)**

---

## ⚙️ Features Used

Final model uses the following features:

* age
* gender
* country
* income
* spendingscore
* days_since_last_purchase

---

## 🤖 Model Used

* **Random Forest Classifier**
* Selected for:

  * High accuracy
  * Handles non-linear data
  * Robust to noise

---

## 📊 Model Performance

* Accuracy: ~XX% (depends on dataset)
* Includes:

  * Confusion Matrix
  * ROC Curve (AUC Score)
  * Feature Importance

---

## 🌐 Streamlit Dashboard

Features:

* 🔮 Customer churn prediction
* 📊 Data visualization dashboard
* 📉 Confusion matrix & ROC curve
* 🧠 Feature importance
* 🎨 Styled UI

---

## ▶️ How to Run

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Train model

```
python model.py
```

### 3. Run app

```
streamlit run app.py
```

---

## 📸 Screenshot

(Add screenshot of dashboard here)

---

## 🚀 Future Improvements

* Deploy with custom domain
* Add real-time data input
* Use advanced models (XGBoost)

---

## 👩‍💻 Author

Tanushree
