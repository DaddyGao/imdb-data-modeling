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
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set plot style
sns.set_theme(style="whitegrid")

# -------------------------------------------------------------
# 1. LOAD & INSPECT DATA
# -------------------------------------------------------------
df = pd.read_csv("insurance.csv")
print("Healthcare Dataset Shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())

# Define predictors and target
numeric_features = ["age", "bmi", "children"]
categorical_features = ["sex", "smoker", "region"]
target = "charges"

X = df[numeric_features + categorical_features]
y = df[target]

# -------------------------------------------------------------
# 2. TRAIN-TEST SPLIT & PREPROCESSING PIPELINE
# -------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_features)
    ]
)

def evaluate_model(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"Model": name, "RMSE ($)": round(rmse, 2), "MAE ($)": round(mae, 2), "R2": round(r2, 4)}

# -------------------------------------------------------------
# 3. MODEL 1: LINEAR REGRESSION (OLS)
# -------------------------------------------------------------
lr_pipe = Pipeline([("prep", preprocessor), ("reg", LinearRegression())])
lr_pipe.fit(X_train, y_train)
lr_res = evaluate_model("1. Linear Regression", y_test, lr_pipe.predict(X_test))

# -------------------------------------------------------------
# 4. MODEL 2: DECISION TREE REGRESSOR (TUNED)
# -------------------------------------------------------------
dt_pipe = Pipeline([("prep", preprocessor), ("reg", DecisionTreeRegressor(random_state=42))])
dt_param_grid = {
    "reg__max_depth": [3, 4, 5, 6, 8],
    "reg__min_samples_split": [5, 10, 20],
    "reg__min_samples_leaf": [2, 5, 10]
}
dt_grid = GridSearchCV(dt_pipe, dt_param_grid, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
dt_grid.fit(X_train, y_train)
best_dt = dt_grid.best_estimator_
dt_res = evaluate_model("2. Decision Tree", y_test, best_dt.predict(X_test))

# -------------------------------------------------------------
# 5. MODEL 3: RANDOM FOREST REGRESSOR (TUNED)
# -------------------------------------------------------------
rf_pipe = Pipeline([("prep", preprocessor), ("reg", RandomForestRegressor(random_state=42))])
rf_param_grid = {
    "reg__n_estimators": [100, 200],
    "reg__max_depth": [4, 6, 8],
    "reg__min_samples_leaf": [2, 4],
    "reg__max_features": [0.6, 0.8, 1.0]
}
rf_grid = GridSearchCV(rf_pipe, rf_param_grid, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_
rf_res = evaluate_model("3. Random Forest", y_test, best_rf.predict(X_test))

# -------------------------------------------------------------
# 6. MODEL 4: SUPPORT VECTOR REGRESSION (RBF KERNEL)
# -------------------------------------------------------------
svr_pipe = Pipeline([("prep", preprocessor), ("reg", SVR(kernel="rbf"))])
svr_param_grid = {
    "reg__C": [1000, 5000, 10000],
    "reg__gamma": ["scale", 0.01, 0.1],
    "reg__epsilon": [10, 50, 100]
}
svr_grid = GridSearchCV(svr_pipe, svr_param_grid, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
svr_grid.fit(X_train, y_train)
best_svr = svr_grid.best_estimator_
svr_res = evaluate_model("4. Support Vector Regression", y_test, best_svr.predict(X_test))

# -------------------------------------------------------------
# 7. SUMMARY TABLE & FEATURE IMPORTANCE PLOT
# -------------------------------------------------------------
results_df = pd.DataFrame([lr_res, dt_res, rf_res, svr_res])
print("\n=== Week 4 Healthcare Application Model Evaluation (Python) ===")
print(results_df.to_string(index=False))
print(f"\nBest Random Forest Params: {rf_grid.best_params_}")

# Feature importance from Random Forest
encoder = best_rf.named_steps["prep"].named_transformers_["cat"]
cat_names = list(encoder.get_feature_names_out(categorical_features))
all_features = numeric_features + cat_names

feat_imp = pd.DataFrame({
    "Feature": all_features,
    "Importance": best_rf.named_steps["reg"].feature_importances_
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(9, 5))
sns.barplot(data=feat_imp, x="Importance", y="Feature", hue="Feature", palette="Blues_r", legend=False)
plt.title("Healthcare Feature Importance Ranking (Random Forest)", fontsize=14)
plt.xlabel("Importance")
plt.ylabel("Predictor")
plt.savefig("python_week4_healthcare_importance.png", dpi=300, bbox_inches="tight")
plt.close()

print("Feature importance plot saved as 'python_week4_healthcare_importance.png'.")