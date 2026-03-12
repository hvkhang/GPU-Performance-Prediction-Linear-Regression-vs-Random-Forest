# GPU Performance Prediction using Machine Learning

This project explores how different machine learning models perform when predicting GPU performance using hardware specifications.

The goal is to compare the effectiveness of:

- **Linear Regression** (baseline model)
- **Ridge Regression** (regularized linear model)
- **Random Forest Regression** (non-linear ensemble model)

The models are evaluated using **cross-validation** to understand how well they generalize to unseen data.

---

# Dataset

The dataset contains specifications for thousands of GPUs, including features such as:

- Memory Bandwidth
- Memory Speed
- Memory Bus Width
- ROPs (Raster Operations)
- Memory Type

These hardware features are used as predictors for the target performance metric.

---

# Modeling Approach

The project compares three regression models with increasing modeling complexity.

## 1. Linear Regression

Linear Regression serves as the **baseline model**.

It assumes a linear relationship between the GPU hardware features and the target variable.

$$
\hat{y} = w^T x + b
$$

Where:
- $x$ is the feature vector
- $w$ are the learned coefficients

Advantages:
- Simple
- Interpretable
- Fast to train

However, Linear Regression can struggle when features are **highly correlated**.

---

## 2. Ridge Regression

Ridge Regression extends Linear Regression by adding **L2 regularization**.

$$
\min_{\beta} ||y - X\beta||^2 + \lambda ||\beta||^2
$$

The regularization term penalizes large coefficients and helps reduce instability caused by correlated features.

Benefits:
- Improves model generalization
- Reduces overfitting
- Stabilizes coefficient estimates

---

## 3. Random Forest Regression

Random Forest is a **tree-based ensemble model** that can capture nonlinear relationships between features.

It works by training multiple decision trees and averaging their predictions.

$$
\hat{y} = \frac{1}{T}\sum_{t=1}^{T} f_t(x)
$$

Advantages:
- Captures nonlinear feature interactions
- Handles complex relationships
- Robust to noise

---

# Model Evaluation

Models are evaluated using **5-fold cross-validation**.

Metrics used:

### Mean Squared Error (MSE)

$$
MSE = \frac{1}{n}\sum (y - \hat{y})^2
$$

### $R^2$ Score

$$
R^2 = 1 - \frac{SS_{res}}{SS_{tot}}
$$

Higher $R^2$ and lower MSE indicate better model performance.

---

# Results Summary

| Model | Mean R² | Mean MSE |
|------|------|------|
| Linear Regression | ... | ... |
| Ridge Regression | ... | ... |
| Random Forest | ... | ... |

In many cases, **Random Forest performs best** because GPU hardware features often interact in nonlinear ways.

---

# Project Structure
gpu-regression-project/
│
├── All_GPUs.csv
├── gpu_regression.ipynb
├── README.md
└── requirements.txt   

---

# Key Skills Demonstrated

- Machine Learning model comparison
- Feature engineering
- Cross-validation
- Regression modeling
- Ensemble methods

---
