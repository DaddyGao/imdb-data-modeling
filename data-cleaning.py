import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual aesthetic
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# -------------------------------------------------------------
# 1. LOAD & INSPECT DATA
# -------------------------------------------------------------
df = pd.read_csv("imdb_top_1000.csv")

print("Initial Shape:", df.shape)
print("\nMissing values before cleaning:\n", df.isnull().sum())

# -------------------------------------------------------------
# 2. DATA CLEANING & FEATURE PARSING
# -------------------------------------------------------------
df_clean = df.copy()

# Deduplicate
df_clean = df_clean.drop_duplicates(subset=["Series_Title", "Released_Year"])

# 1. Clean Released_Year (coerce non-numeric strings to NaN, then convert)
df_clean["Released_Year"] = pd.to_numeric(df_clean["Released_Year"], errors="coerce")
df_clean = df_clean.dropna(subset=["Released_Year"])
df_clean["Released_Year"] = df_clean["Released_Year"].astype(int)

# 2. Clean Runtime: remove ' min' suffix and convert to float
df_clean["Runtime_min"] = df_clean["Runtime"].astype(str).str.replace(" min", "").astype(float)

# 3. Clean Gross: remove commas and parse to float
df_clean["Gross_clean"] = df_clean["Gross"].astype(str).str.replace(",", "").replace("nan", np.nan).astype(float)

# 4. Extract Primary Genre (first token before comma)
df_clean["Primary_Genre"] = df_clean["Genre"].apply(lambda x: str(x).split(",")[0].strip() if pd.notnull(x) else "Unknown")

# 5. Handle Missing Values without chained inplace assignment
df_clean["Meta_score"] = df_clean["Meta_score"].fillna(df_clean["Meta_score"].median())
df_clean["Gross_clean"] = df_clean["Gross_clean"].fillna(df_clean["Gross_clean"].median())
df_clean["Certificate"] = df_clean["Certificate"].fillna("Unknown")

# 6. Log-transform skewed numeric features (exact column casing: No_of_Votes)
df_clean["No_of_Votes"] = pd.to_numeric(df_clean["No_of_Votes"], errors="coerce")
df_clean["log_No_of_Votes"] = np.log1p(df_clean["No_of_Votes"])
df_clean["log_Gross"] = np.log1p(df_clean["Gross_clean"])

# Filter final modeling dataset
feature_cols = [
    "Released_Year", "Runtime_min", "Meta_score", "No_of_Votes",
    "Gross_clean", "log_No_of_Votes", "log_Gross", "Primary_Genre",
    "Certificate", "Director", "Star1", "IMDB_Rating"
]
df_model_ready = df_clean[feature_cols].copy()

print("\nFinal Cleaned Shape:", df_model_ready.shape)

# -------------------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS & VISUALIZATIONS
# -------------------------------------------------------------

# Visual 1: Distribution of IMDB Ratings
plt.figure(figsize=(8, 5))
sns.histplot(df_model_ready["IMDB_Rating"], bins=20, kde=True, color="#1f77b4")
plt.title("Distribution of IMDb Top 1000 Ratings", fontsize=14)
plt.xlabel("IMDb Rating")
plt.ylabel("Movie Count")
plt.savefig("python_eda_target_dist.png", dpi=300, bbox_inches="tight")
plt.close()

# Visual 2: Correlation Matrix of Key Numeric Features
plt.figure(figsize=(8, 6))
numeric_cols = ["IMDB_Rating", "Meta_score", "Runtime_min", "log_No_of_Votes", "log_Gross", "Released_Year"]
corr_mat = df_model_ready[numeric_cols].corr()
sns.heatmap(corr_mat, annot=True, cmap="mako", fmt=".2f", square=True)
plt.title("Correlation Matrix of Numeric Predictors", fontsize=14)
plt.savefig("python_eda_correlation_matrix.png", dpi=300, bbox_inches="tight")
plt.close()

# Visual 3: IMDB Rating across Top 8 Primary Genres
top_genres = df_model_ready["Primary_Genre"].value_counts().nlargest(8).index
plt.figure(figsize=(10, 5))
sns.boxplot(
    data=df_model_ready[df_model_ready["Primary_Genre"].isin(top_genres)],
    x="Primary_Genre",
    y="IMDB_Rating",
    hue="Primary_Genre",
    palette="Spectral",
    legend=False
)
plt.title("IMDb Rating by Primary Genre (Top 8 Genres)", fontsize=14)
plt.xlabel("Primary Genre")
plt.ylabel("IMDb Rating")
plt.xticks(rotation=25)
plt.savefig("python_eda_genre_boxplot.png", dpi=300, bbox_inches="tight")
plt.close()

# Export clean CSV for modeling
df_model_ready.to_csv("imdb_top_1000_cleaned_py.csv", index=False)
print("\nExported clean data to 'imdb_top_1000_cleaned_py.csv' and saved 3 EDA plots.")