import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

#! === 1. Load dataset ===
All_GPUs = pd.read_csv("ALL_GPUS.csv")

#! === 2. Clean missing values ===
All_GPUs = All_GPUs.replace("", np.nan)
All_GPUs = All_GPUs.replace("^\\n- $", np.nan, regex=True)
All_GPUs = All_GPUs.replace("NA", np.nan)

#! === 3. Missing data summary ===
NA_table = pd.DataFrame({
    "Column": All_GPUs.columns,
    "NA_Count": All_GPUs.isna().sum().values,
    "NA_Percentage": All_GPUs.isna().mean().values * 100
})
NA_table = NA_table.sort_values("NA_Percentage")

# Keep columns with < 16% missing values, then drop rows with NA
keep_columns = NA_table[NA_table["NA_Percentage"] < 16]["Column"]
new_All_GPUs = All_GPUs[keep_columns].dropna()

#! === 4. Functions to clean units and process ROPs ===
def remove_units(series):
    return pd.to_numeric(series.astype(str).str.replace(r"[^0-9.]", "", regex=True), errors="coerce")

# Process expressions into numeric values for ROPs
def ROPs_proc(series):
    series = series.astype(str)
    number = series.str.extract(r"(\d+)").astype(float)
    multiplier = series.str.extract(r"\(x(\d+)\)").astype(float)
    multiplier = multiplier.fillna(1)
    return (number[0] * multiplier[0]).astype(float)

#! === 5. Apply cleaning ===
cols_to_clean = ["Memory_Bandwidth", "Memory_Bus", "Memory_Speed"]
for col in cols_to_clean:
    new_All_GPUs[col] = remove_units(new_All_GPUs[col])

new_All_GPUs["ROPs"] = ROPs_proc(new_All_GPUs["ROPs"])

#! === 6. Select main variables ===
main_data = new_All_GPUs[["Memory_Bandwidth", "Memory_Speed", "Memory_Bus", "ROPs", "Memory_Type"]].copy()

# Encode Memory_Type as numeric
main_data["Memory_Type_Numeric"] = pd.factorize(main_data["Memory_Type"])[0] + 1

#! === 7. Log-transform selected columns ===
log_cols = ["Memory_Bandwidth", "Memory_Bus", "Memory_Speed", "ROPs"]
for col in log_cols:
    main_data[col] = np.log(main_data[col].astype(float))

#! === 8. Summary statistics ===
numeric_data = main_data.select_dtypes(include=[np.number])

summary_stats = pd.DataFrame({
    "Mean": numeric_data.mean(),
    "SD": numeric_data.std(),
    "Min": numeric_data.min(),
    "Q1": numeric_data.quantile(0.25),
    "Median": numeric_data.median(),
    "Q3": numeric_data.quantile(0.75),
    "Max": numeric_data.max()
})

print(summary_stats)

#! === 9. Visualization examples (hist, boxplot, scatter) ===
# sns.histplot(main_data["Memory_Bandwidth"], bins=30, color="#77C7B9")
# plt.title("Distribution of Memory Bandwidth (log scale)")
# plt.xlabel("Memory Bandwidth (log scale, GB/s)")
# plt.ylabel("Count")
# plt.show()

# sns.boxplot(x="Memory_Type", y="Memory_Bandwidth", data=main_data, color="#77C7B9")
# plt.title("Boxplot of Memory Bandwidth by Memory Type")
# plt.show()

# sns.scatterplot(x="Memory_Speed", y="Memory_Bandwidth", data=main_data, color="#77C7B9")
# plt.title("Scatter: Memory Bandwidth vs Memory Speed")
# plt.show()

#! === 10. Train/test split ===
X = main_data[["Memory_Bus", "Memory_Speed", "ROPs", "Memory_Type_Numeric"]]
y = main_data["Memory_Bandwidth"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

#! === 11. Linear regression model ===
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

#! === 12. Statsmodels regression summary ===
X_train_const = sm.add_constant(X_train)
ols_model = sm.OLS(y_train, X_train_const).fit()
print(ols_model.summary())

#! === 13. Evaluation ===
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)
print(f"RMSE: {rmse}")
print(f"R-squared: {r2}")