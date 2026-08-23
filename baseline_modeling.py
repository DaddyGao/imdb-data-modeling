import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -------------------------------------------------------------
# 1. LOAD CLEANED DATA
# -------------------------------------------------------------
df = pd.read_csv("imdb_top_1000_cleaned_py.csv")

# Define feature sets
numeric_features = ["Released_Year", "Runtime_min", "Meta_score", "log_No_of_Votes", "log_Gross"]
categorical_features = ["Primary_Genre", "Certificate"]
target = "IMDB_Rating"

X = df[numeric_features + categorical_features]
y = df[target]

# -------------------------------------------------------------
# 2. TRAIN-TEST SPLIT & PREPROCESSING PIPELINE
# -------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Preprocessor: Scale numeric, One-Hot encode categorical
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
    ]
)

# Helper function to evaluate and return metrics
def evaluate_model(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"Model": name, "RMSE": round(rmse, 4), "MAE": round(mae, 4), "R2": round(r2, 4)}

# -------------------------------------------------------------
# 3. MODEL 1: LINEAR REGRESSION (BASELINE)
# -------------------------------------------------------------
lr_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

lr_pipeline.fit(X_train, y_train)
y_pred_lr = lr_pipeline.predict(X_test)
lr_metrics = evaluate_model("Linear Regression (Python)", y_test, y_pred_lr)

# -------------------------------------------------------------
# 4. MODEL 2: DECISION TREE REGRESSOR (WITH CV TUNING)
# -------------------------------------------------------------
dt_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", DecisionTreeRegressor(random_state=42))
])

param_grid_dt = {
    "regressor__max_depth": [3, 5, 7, 10, 15],
    "regressor__min_samples_split": [2, 5, 10],
    "regressor__min_samples_leaf": [1, 2, 4]
}

dt_grid = GridSearchCV(
    dt_pipeline,
    param_grid_dt,
    cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1
)
dt_grid.fit(X_train, y_train)

best_dt = dt_grid.best_estimator_
y_pred_dt = best_dt.predict(X_test)
dt_metrics = evaluate_model("Decision Tree Regressor (Python)", y_test, y_pred_dt)

# -------------------------------------------------------------
# 5. SUMMARY RESULTS & RESIDUAL VISUALIZATION
# -------------------------------------------------------------
results_df = pd.DataFrame([lr_metrics, dt_metrics])
print("\n=== Week 2 Python Model Evaluation on Test Set (80/20) ===")
print(results_df.to_string(index=False))
print(f"\nBest Decision Tree Parameters: {dt_grid.best_params_}")

# Residual Comparison Plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
residuals_lr = y_test - y_pred_lr
sns.scatterplot(x=y_pred_lr, y=residuals_lr, alpha=0.7, color="navy")
plt.axhline(0, color="red", linestyle="--")
plt.title("Linear Regression: Residuals vs. Fitted")
plt.xlabel("Predicted IMDb Rating")
plt.ylabel("Residuals")

plt.subplot(1, 2, 2)
residuals_dt = y_test - y_pred_dt
sns.scatterplot(x=y_pred_dt, y=residuals_dt, alpha=0.7, color="darkgreen")
plt.axhline(0, color="red", linestyle="--")
plt.title("Tuned Decision Tree: Residuals vs. Fitted")
plt.xlabel("Predicted IMDb Rating")
plt.ylabel("Residuals")

plt.tight_layout()
plt.savefig("python_week2_residuals.png", dpi=300)
plt.close()