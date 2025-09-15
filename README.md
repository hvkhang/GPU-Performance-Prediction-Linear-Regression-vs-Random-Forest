# Linear Regression on GPU Dataset

This project demonstrates a **linear regression pipeline in Python** using `scikit-learn`, and `statsmodels`.  
The dataset consists of GPU specifications, and the goal is to model the relationship between **memory features** (e.g., bus width, speed, type) and **memory bandwidth**.

---

## Features
- Data cleaning and preprocessing:
  - Handle missing values.
  - Remove units from numeric columns.
  - Transform categorical variables.
- Feature engineering:
  - Apply log transformation.
  - Create summary statistics.
- Data visualization with `matplotlib` and `seaborn`.
- Multiple Linear Regression (MLR) using:
  - **Scikit-learn** (prediction & evaluation).
  - **Statsmodels** (detailed statistical summary).
- Model evaluation with:
  - Root Mean Square Error (RMSE).
  - R-squared (R²).

---

## Project Structure
```
LR_Python/
│── gpu_linear_regression.py   # Main Python script
│── requirements.txt           # Required dependencies
│── README.md                  # Project documentation
│── data/                      # (Optional) dataset folder
```

---

## Installation
Clone the repository and install the required libraries:
```bash
git clone https://github.com/YOUR_USERNAME/linear-regression-python.git
cd linear-regression-python
pip install -r requirements.txt
```

---

## Usage
Run the pipeline:
```bash
python gpu_linear_regression.py
```

## ✨ Author
- **Khang-Huynh Vuong** – Ho Chi Minh City University of Technology - Faculty of Computer Science and Engineering.  
