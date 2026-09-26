# 📘 Topic 01: Supervised Learning

Supervised Learning is a branch of Machine Learning where models are trained using labeled data. The dataset contains input features ($X$) and corresponding target labels ($y$). The objective is to learn a mapping function $f(X) \rightarrow y$ to accurately predict outcomes for new, unseen data.

---

## 💡 Core Concepts

### 1. Types of Supervised Learning
* **Regression**: Used when the target variable ($y$) is continuous (e.g., house prices, stock values, temperature).
* **Classification**: Used when the target variable ($y$) is categorical (e.g., Spam vs. Not Spam, Loan Approved vs. Rejected).

### 2. Model Performance & Tradeoffs
* **Bias-Variance Tradeoff**:
  * **High Bias (Underfitting)**: Model is too simple to capture underlying patterns in data.
  * **High Variance (Overfitting)**: Model learns noise and training details too well, failing to generalize to test data.
* **Loss Functions**: Functions used during training to measure prediction errors (e.g., Mean Squared Error for regression, Binary Cross-Entropy for classification).
* **Backpropagation**: An optimization process using gradient descent to update weights and minimize loss.

---

## 🛠️ Supervised Learning Algorithms

| Algorithm | Type | Description / Best Use Case | Key Hyperparameters |
| :--- | :---: | :--- | :--- |
| **Linear Regression** | Regression | Models linear relationships between inputs and target. | Fit intercept, Regularization ($\alpha$) |
| **Logistic Regression** | Classification | Uses Sigmoid function to output class probabilities. | Regularization ($C$, penalty) |
| **K-Nearest Neighbors (KNN)** | Both | Classifies data points based on proximity to nearest $k$ neighbors. | $n\_neighbors$, metric (Euclidean, Manhattan) |
| **Support Vector Machine (SVM)** | Both | Finds optimal hyperplane maximizing margin between classes. | $C$, kernel (Linear, RBF, Poly) |
| **Decision Tree** | Both | Tree structure splitting data on feature thresholds. | $max\_depth$, $min\_samples\_split$ |
| **Random Forest** | Both | Ensemble of decision trees trained on bootstrapped data splits. | $n\_estimators$, $max\_depth$, $max\_features$ |

---

## 📊 Evaluation Metrics

### Regression Metrics
* **Mean Squared Error (MSE)**: Average squared difference between actual and predicted values.
* **Root Mean Squared Error (RMSE)**: Square root of MSE, expressed in original target units.
* **$R^2$ Score (Coefficient of Determination)**: Proportion of variance in target variable explained by model ($0 \rightarrow 1$).

### Classification Metrics
* **Confusion Matrix**: Table comparing True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN).
* **Accuracy**: $\frac{TP + TN}{TP + TN + FP + FN}$
* **Precision**: $\frac{TP}{TP + FP}$ *(Crucial when minimizing False Positives)*
* **Recall (Sensitivity)**: $\frac{TP}{TP + FN}$ *(Crucial when minimizing False Negatives)*
* **F1-Score**: Harmonic mean of Precision and Recall: $2 \times \frac{Precision \times Recall}{Precision + Recall}$

---

## 💻 Python Implementation Example

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix

# -------------------------------------------------------------
# 1. REGRESSION DEMONSTRATION (House Price Prediction)
# -------------------------------------------------------------
# Features: [Square Footage, Bedrooms] -> Target: Price ($k)
X_reg = np.array([[1200, 2], [1500, 3], [1800, 3], [2400, 4], [3000, 5], [3500, 5]])
y_reg = np.array([250, 310, 360, 480, 600, 710])

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)

reg_model = LinearRegression()
reg_model.fit(X_train_r, y_train_r)
y_pred_r = reg_model.predict(X_test_r)

print("--- REGRESSION RESULTS ---")
print(f"Test MSE: {mean_squared_error(y_test_r, y_pred_r):.2f}")
print(f"R2 Score: {r2_score(y_test_r, y_pred_r):.2f}")

# -------------------------------------------------------------
# 2. CLASSIFICATION DEMONSTRATION (Loan Approval)
# -------------------------------------------------------------
# Features: [Credit Score, Monthly Income ($k)] -> Target: Approved (1) / Rejected (0)
X_cls = np.array([[600, 3.5], [750, 8.0], [580, 2.8], [710, 6.5], [800, 10.0], [620, 3.0]])
y_cls = np.array([0, 1, 0, 1, 1, 0])

clf_model = LogisticRegression()
clf_model.fit(X_cls, y_cls)
y_pred_c = clf_model.predict(X_cls)

print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(y_cls, y_pred_c, target_names=['Rejected', 'Approved']))
```

---

## 🎯 Practical Exercise Assignment

### Task Title: "The Smart Real-Estate & Loan Approval Engine"

#### Part 1: Regression Task (Real-Estate Price Predictor)
* **Dataset:** Kaggle's Boston Housing or California Housing dataset.
* **Goal:** Use **Linear Regression** and **Decision Tree Regressor** to predict house prices.
* **Deliverables:**
  1. Compare Train vs. Test MSE and $R^2$ scores across both models.
  2. Perform overfitting analysis on Decision Tree Regressor by tuning `max_depth`.

#### Part 2: Classification Task (Loan Approval Predictor)
* **Dataset:** Loan Prediction Dataset (Analytics Vidhya / Kaggle).
* **Goal:** Classify whether a customer will receive Loan Approval (`1`) or Rejection (`0`).
* **Deliverables:**
  1. Train and compare **Logistic Regression**, **KNN**, and **Random Forest** models.
  2. Generate Confusion Matrix, Precision, Recall, and F1-Score reports.
  3. Identify which model yields the lowest False Positive rate.
