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

# -------------------------------------------------------------
# 1. LOAD CLEANED DATA
# -------------------------------------------------------------
df = pd.read_csv("imdb_top_1000_cleaned_py.csv")

numeric_features = ["Released_Year", "Runtime_min", "Meta_score", "log_No_of_Votes", "log_Gross"]
categorical_features = ["Primary_Genre", "Certificate"]
target = "IMDB_Rating"

X = df[numeric_features + categorical_features]
y = df[target]

# 80/20 train-test split (identical seed to Week 2)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
    ]
)

def evaluate_model(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"Model": name, "RMSE": round(rmse, 4), "MAE": round(mae, 4), "R2": round(r2, 4)}

# -------------------------------------------------------------
# 2. RUN BASELINES (WEEK 2 REFERENCE)
# -------------------------------------------------------------
# Linear Regression
lr_pipe = Pipeline([("prep", preprocessor), ("reg", LinearRegression())])
lr_pipe.fit(X_train, y_train)
lr_metrics = evaluate_model("1. Linear Regression", y_test, lr_pipe.predict(X_test))

# Decision Tree (Tuned)
dt_pipe = Pipeline([("prep", preprocessor), ("reg", DecisionTreeRegressor(random_state=42, max_depth=7, min_samples_split=10, min_samples_leaf=4))])
dt_pipe.fit(X_train, y_train)
dt_metrics = evaluate_model("2. Decision Tree", y_test, dt_pipe.predict(X_test))

# -------------------------------------------------------------
# 3. MODEL 3: RANDOM FOREST REGRESSOR (TUNED)
# -------------------------------------------------------------
rf_pipe = Pipeline([
    ("prep", preprocessor),
    ("reg", RandomForestRegressor(random_state=42))
])

rf_param_grid = {
    "reg__n_estimators": [100, 200],
    "reg__max_depth": [6, 8, 12],
    "reg__min_samples_leaf": [2, 4],
    "reg__max_features": ["sqrt", 0.5]
}

rf_grid = GridSearchCV(rf_pipe, rf_param_grid, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_

y_pred_rf = best_rf.predict(X_test)
rf_metrics = evaluate_model("3. Random Forest", y_test, y_pred_rf)

# -------------------------------------------------------------
# 4. MODEL 4: SUPPORT VECTOR REGRESSION (RBF KERNEL)
# -------------------------------------------------------------
svr_pipe = Pipeline([
    ("prep", preprocessor),
    ("reg", SVR(kernel="rbf"))
])

svr_param_grid = {
    "reg__C": [0.5, 1.0, 2.0, 5.0],
    "reg__epsilon": [0.01, 0.05, 0.1],
    "reg__gamma": ["scale", "auto", 0.01, 0.1]
}

svr_grid = GridSearchCV(svr_pipe, svr_param_grid, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
svr_grid.fit(X_train, y_train)
best_svr = svr_grid.best_estimator_

y_pred_svr = best_svr.predict(X_test)
svr_metrics = evaluate_model("4. Support Vector Regression", y_test, y_pred_svr)

# -------------------------------------------------------------
# 5. COMPARISON TABLE & FEATURE IMPORTANCE
# -------------------------------------------------------------
comparison_df = pd.DataFrame([lr_metrics, dt_metrics, rf_metrics, svr_metrics])
print("\n=== Comprehensive Model Evaluation Matrix (Python) ===")
print(comparison_df.to_string(index=False))
print(f"\nBest Random Forest Hyperparameters: {rf_grid.best_params_}")
print(f"Best SVR Hyperparameters: {svr_grid.best_params_}")

# Extract and plot Random Forest feature importances
encoder = best_rf.named_steps["prep"].named_transformers_["cat"]
cat_encoded_names = list(encoder.get_feature_names_out(categorical_features))
feature_names = numeric_features + cat_encoded_names

importances = best_rf.named_steps["reg"].feature_importances_
feat_imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
feat_imp_df = feat_imp_df.sort_values(by="Importance", ascending=False).head(10)

plt.figure(figsize=(9, 5))
sns.barplot(data=feat_imp_df, x="Importance", y="Feature", hue="Feature", palette="viridis", legend=False)
plt.title("Top 10 Feature Importances (Random Forest)", fontsize=14)
plt.xlabel("Gini / Variance Reduction Importance")
plt.ylabel("Predictor Variable")
plt.savefig("python_week3_rf_feature_importance.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nFeature importance plot saved as 'python_week3_rf_feature_importance.png'.")