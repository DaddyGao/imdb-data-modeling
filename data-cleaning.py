import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# -------------------------------------------------------------
# 1. LOAD & INSPECT DATA
# -------------------------------------------------------------
# Using Kaggle IMDb 5000 / metadata standard schema
df = pd.read_csv("movie_metadata.csv")

print("Initial Shape:", df.shape)
print("\nMissing Values per Column:\n", df.isnull().sum()[df.isnull().sum() > 0])

# -------------------------------------------------------------
# 2. DATA CLEANING & PREPARATION
# -------------------------------------------------------------
# Select relevant predictor columns and target (imdb_score)
selected_cols = [
    "movie_title", "title_year", "duration", "budget", "gross",
    "num_voted_users", "num_critic_for_reviews", "director_facebook_likes",
    "cast_total_facebook_likes", "genres", "imdb_score"
]
df_clean = df[selected_cols].copy()

# Remove duplicates based on movie title and release year
df_clean.drop_duplicates(subset=["movie_title", "title_year"], inplace=True)

# Drop rows missing the target variable
df_clean.dropna(subset=["imdb_score"], inplace=True)

# Impute numeric missing values with median
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if df_clean[col].isnull().sum() > 0:
        df_clean[col].fillna(df_clean[col].median(), inplace=True)

# Feature Engineering: Extract primary genre and log-transform skewed financials
df_clean["primary_genre"] = df_clean["genres"].apply(lambda x: str(x).split("|")[0] if pd.notnull(x) else "Unknown")
df_clean["log_budget"] = np.log1p(df_clean["budget"])
df_clean["log_gross"] = np.log1p(df_clean["gross"])

print("Cleaned Dataset Shape:", df_clean.shape)

# -------------------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS & VISUALIZATION
# -------------------------------------------------------------
# Visual 1: Target Variable Distribution (IMDb Rating)
plt.figure(figsize=(8, 5))
sns.histplot(df_clean["imdb_score"], bins=30, kde=True, color="royalblue")
plt.title("Distribution of IMDb Scores")
plt.xlabel("IMDb Score (1-10)")
plt.ylabel("Frequency")
plt.savefig("python_eda_target_distribution.png", dpi=300, bbox_inches="tight")
plt.show()

# Visual 2: Correlation Heatmap of Numeric Predictors
plt.figure(figsize=(9, 7))
corr_matrix = df_clean[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
plt.title("Correlation Matrix of Numeric Features")
plt.savefig("python_eda_correlation_matrix.png", dpi=300, bbox_inches="tight")
plt.show()

# Visual 3: Top Genres vs. IMDb Score Boxplot
top_genres = df_clean["primary_genre"].value_counts().nlargest(8).index
plt.figure(figsize=(10, 6))
sns.boxplot(
    data=df_clean[df_clean["primary_genre"].isin(top_genres)],
    x="primary_genre",
    y="imdb_score",
    palette="Set2"
)
plt.title("IMDb Score Distribution by Primary Genre (Top 8 Genres)")
plt.xlabel("Genre")
plt.ylabel("IMDb Score")
plt.xticks(rotation=30)
plt.savefig("python_eda_genre_boxplot.png", dpi=300, bbox_inches="tight")
plt.show()

# Export clean data for modeling phase
df_clean.to_csv("imdb_cleaned_week1.csv", index=False)