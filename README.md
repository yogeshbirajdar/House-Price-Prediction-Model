# 🏠 House Price Prediction Web App

A Machine Learning web application built using **Streamlit** that predicts house prices based on multiple property-related features.

This project demonstrates the complete ML lifecycle:
- Data preprocessing
- Feature engineering
- Polynomial transformation
- Model training using Lasso Regression
- Model serialization using Pickle
- Web app deployment using Streamlit
- Cloud deployment using GitHub & Streamlit Cloud

---

# 📌 Project Overview

The goal of this project is to predict house prices based on various features such as:

- Crime Rate
- Plot Area
- Industrial Area Distance
- Near River
- Pollution Level
- Number of Rooms
- House Age
- Distance to City
- Highway Access
- Property Tax
- Distance to School
- Population Factor
- Low Income Percentage

The model takes these inputs and predicts the estimated house price in USD.

---

# 🧠 Machine Learning Theory Behind the Project

## 1️⃣ Problem Type

This is a **Supervised Machine Learning Regression Problem**.

Why Regression?
Because the output (house price) is a continuous numerical value.

---

## 2️⃣ Data Preprocessing

Before training the model, we performed:

- Handling missing values (if any)
- Feature selection
- Log transformation (to handle skewness)
- Polynomial feature transformation

---

## 3️⃣ Polynomial Feature Transformation

We used **PolynomialFeatures** from Scikit-Learn.

Why?

Because:
- Real-world relationships are not always linear
- Polynomial features help capture complex relationships
- It improves model performance by adding interaction terms

Example:
If we have:
X1 and X2

Polynomial transformation creates:
- X1²
- X2²
- X1 × X2

This helps model understand feature interactions.

---

## 4️⃣ Model Used: Lasso Regression

We used **Lasso Regression**.

Why Lasso?

- It performs feature selection
- It reduces overfitting
- It handles multicollinearity
- It adds regularization (L1 penalty)

Lasso Formula:

Loss = RSS + λ * (Sum of absolute coefficients)

Where:
- RSS = Residual Sum of Squares
- λ = Regularization parameter

Lasso automatically removes less important features by shrinking their coefficients to zero.

---

## 5️⃣ Model Evaluation

We evaluated the model using:

- R² Score (Coefficient of Determination)

R² tells us how well the model explains variance in data.

Example Results:
- Train R² ≈ 0.90+
- Test R² ≈ 0.80+

This indicates good generalization with controlled overfitting.

---

# 🏗️ Project Workflow (Step-by-Step)

1. Data Collection
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. Log Transformation
5. Polynomial Feature Engineering
6. Train-Test Split
7. Model Training (Lasso Regression)
8. Model Evaluation
9. Model Saving using Pickle
10. Web App Creation using Streamlit
11. UI Styling using HTML & CSS
12. Deployment using GitHub
13. Deployment on Streamlit Cloud

---

# 🗂️ Project Structure
