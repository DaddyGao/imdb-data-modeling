library(dplyr)
library(stringr)
library(ggplot2)
library(corrplot)

# -------------------------------------------------------------
# 1. LOAD DATA
# -------------------------------------------------------------
df <- read.csv("imdb_top_1000.csv", stringsAsFactors = FALSE)

cat("Initial Dimensions:", dim(df), "\n")

# -------------------------------------------------------------
# 2. DATA CLEANING & PREPARATION
# -------------------------------------------------------------
df_clean <- df %>%
  # Deduplicate
  distinct(Series_Title, Released_Year, .keep_all = TRUE) %>%
  # Fix non-numeric Released_Year values
  mutate(Released_Year = suppressWarnings(as.numeric(Released_Year))) %>%
  filter(!is.na(Released_Year)) %>%
  # Clean Runtime and Gross string formatting
  mutate(
    Runtime_min = as.numeric(str_replace(Runtime, " min", "")),
    Gross_clean = as.numeric(str_replace_all(Gross, ",", "")),
    Primary_Genre = str_trim(sapply(strsplit(as.character(Genre), ","), `[`, 1)),
    Certificate = ifelse(is.na(Certificate) | Certificate == "", "Unknown", Certificate)
  ) %>%
  # Median Imputation for numeric features
  mutate(
    Meta_score = ifelse(is.na(Meta_score), median(Meta_score, na.rm = TRUE), Meta_score),
    Gross_clean = ifelse(is.na(Gross_clean), median(Gross_clean, na.rm = TRUE), Gross_clean)
  ) %>%
  # Log-transforms for skewed predictors
  mutate(
    log_No_of_votes = log1p(No_of_votes),
    log_Gross = log1p(Gross_clean)
  ) %>%
  select(
    Released_Year, Runtime_min, Meta_score, No_of_votes,
    Gross_clean, log_No_of_votes, log_Gross, Primary_Genre,
    Certificate, Director, Star1, IMDB_Rating
  )

cat("Cleaned Dimensions:", dim(df_clean), "\n")

# -------------------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS & VISUALIZATIONS
# -------------------------------------------------------------

# Visual 1: Target Variable Distribution
p1 <- ggplot(df_clean, aes(x = IMDB_Rating)) +
  geom_histogram(aes(y = after_stat(density)), bins = 20, fill = "#1f77b4", color = "white") +
  geom_density(color = "#d62728", linewidth = 1) +
  theme_minimal() +
  labs(
    title = "Distribution of IMDb Top 1000 Ratings",
    x = "IMDb Rating",
    y = "Density"
  )
print(p1)
ggsave("r_eda_target_dist.png", plot = p1, width = 8, height = 5, dpi = 300)

# Visual 2: Correlation Matrix
numeric_cols <- df_clean %>%
  select(IMDB_Rating, Meta_score, Runtime_min, log_No_of_votes, log_Gross, Released_Year)

corr_matrix <- cor(numeric_cols, use = "complete.obs")

png("r_eda_correlation_matrix.png", width = 800, height = 700)
corrplot(
  corr_matrix,
  method = "circle",
  type = "upper",
  addCoef.col = "black",
  tl.col = "black",
  number.cex = 0.8,
  title = "Correlation Matrix of Numeric Predictors",
  mar = c(0, 0, 1, 0)
)
dev.off()

# Visual 3: Primary Genre vs. IMDB Rating
top_genres <- names(sort(table(df_clean$Primary_Genre), decreasing = TRUE)[1:8])

p3 <- df_clean %>%
  filter(Primary_Genre %in% top_genres) %>%
  ggplot(aes(x = reorder(Primary_Genre, IMDB_Rating, median), y = IMDB_Rating, fill = Primary_Genre)) +
  geom_boxplot(show.legend = FALSE, alpha = 0.8) +
  theme_minimal() +
  coord_flip() +
  labs(
    title = "IMDb Rating by Primary Genre (Top 8 Genres)",
    x = "Primary Genre",
    y = "IMDb Rating"
  )
print(p3)
ggsave("r_eda_genre_boxplot.png", plot = p3, width = 8, height = 5, dpi = 300)

# Export clean CSV
write.csv(df_clean, "imdb_top_1000_cleaned_r.csv", row.names = FALSE)