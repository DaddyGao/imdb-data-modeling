# Install packages if not already installed:
# install.packages(c("dplyr", "stringr", "caret", "rpart", "ggplot2", "e1071"))

library(dplyr)
library(stringr)
library(caret)
library(rpart)
library(ggplot2)

# -------------------------------------------------------------
# 1. LOAD & CLEAN DATA INLINE
# -------------------------------------------------------------
raw_df <- read.csv("/Users/sylvain/Sofia/MSCS3807 Data Modeling in Python and R/imdb-data-modeling/imdb_top_1000.csv", stringsAsFactors = FALSE)

# Preprocessing and Feature Engineering
df_clean <- raw_df %>%
  distinct(Series_Title, Released_Year, .keep_all = TRUE) %>%
  mutate(Released_Year = suppressWarnings(as.numeric(Released_Year))) %>%
  filter(!is.na(Released_Year)) %>%
  mutate(
    Runtime_min = as.numeric(str_replace(Runtime, " min", "")),
    Gross_clean = as.numeric(str_replace_all(Gross, ",", "")),
    Primary_Genre = str_trim(sapply(strsplit(as.character(Genre), ","), `[`, 1)),
    Certificate = ifelse(is.na(Certificate) | Certificate == "", "Unknown", Certificate)
  ) %>%
  mutate(
    Meta_score = ifelse(is.na(Meta_score), median(Meta_score, na.rm = TRUE), Meta_score),
    Gross_clean = ifelse(is.na(Gross_clean), median(Gross_clean, na.rm = TRUE), Gross_clean)
  ) %>%
  mutate(
    log_No_of_Votes = log1p(as.numeric(No_of_Votes)),
    log_Gross = log1p(Gross_clean)
  )

# Select modeling features and convert categoricals to factors
# Lump rare categories (< 15 occurrences) into "Other"
model_data <- df_clean %>%
  select(
    Released_Year, Runtime_min, Meta_score, log_No_of_Votes, log_Gross,
    Primary_Genre, Certificate, IMDB_Rating
  ) %>%
  mutate(
    Primary_Genre = forcats::fct_lump_min(Primary_Genre, min = 15, other_level = "Other"),
    Certificate = forcats::fct_lump_min(Certificate, min = 15, other_level = "Other")
  )

# -------------------------------------------------------------
# 2. TRAIN-TEST SPLIT & 5-FOLD CROSS-VALIDATION
# -------------------------------------------------------------
set.seed(42)
train_index <- createDataPartition(model_data$IMDB_Rating, p = 0.80, list = FALSE)
train_set <- model_data[train_index, ]
test_set  <- model_data[-train_index, ]

ctrl <- trainControl(method = "cv", number = 5)

calc_metrics <- function(name, y_true, y_pred) {
  rmse_val <- RMSE(y_pred, y_true)
  mae_val  <- MAE(y_pred, y_true)
  r2_val   <- R2(y_pred, y_true)
  data.frame(
    Model = name,
    RMSE = round(rmse_val, 4),
    MAE = round(mae_val, 4),
    R2 = round(r2_val, 4)
  )
}

# -------------------------------------------------------------
# 3. MODEL 1: LINEAR REGRESSION (BASELINE OLS)
# -------------------------------------------------------------
set.seed(42)
# Add "zv" (zero-variance removal) to preProcess
lr_model <- train(
  IMDB_Rating ~ .,
  data = train_set,
  method = "lm",
  preProcess = c("zv", "center", "scale"),
  trControl = ctrl
)

pred_lr <- predict(lr_model, newdata = test_set)
lr_res <- calc_metrics("Linear Regression (R caret)", test_set$IMDB_Rating, pred_lr)

# -------------------------------------------------------------
# 4. MODEL 2: DECISION TREE (rpart TUNING)
# -------------------------------------------------------------
set.seed(42)
dt_grid <- expand.grid(cp = seq(0.001, 0.05, by = 0.005))

dt_model <- train(
  IMDB_Rating ~ .,
  data = train_set,
  method = "rpart",
  tuneGrid = dt_grid,
  trControl = ctrl
)

pred_dt <- predict(dt_model, newdata = test_set)
dt_res <- calc_metrics("Decision Tree (R caret)", test_set$IMDB_Rating, pred_dt)

# -------------------------------------------------------------
# 5. SUMMARY RESULTS & RESIDUAL PLOT
# -------------------------------------------------------------
summary_results <- rbind(lr_res, dt_res)
cat("\n=== Week 2 R Model Evaluation on Test Set (80/20) ===\n")
print(summary_results)

test_set$Pred_LR <- pred_lr
test_set$Resid_LR <- test_set$IMDB_Rating - pred_lr

p_res <- ggplot(test_set, aes(x = Pred_LR, y = Resid_LR)) +
  geom_point(color = "navy", alpha = 0.6) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  theme_minimal() +
  labs(
    title = "Linear Regression (R): Residuals vs Fitted",
    x = "Predicted IMDb Rating",
    y = "Residuals"
  )

ggsave("r_week2_residuals.png", plot = p_res, width = 7, height = 5, dpi = 300)