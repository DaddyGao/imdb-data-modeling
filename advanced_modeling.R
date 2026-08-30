library(dplyr)
library(stringr)
library(caret)
library(ggplot2)
library(randomForest)
library(e1071)

# -------------------------------------------------------------
# 1. LOAD & CLEAN DATA
# -------------------------------------------------------------
raw_df <- read.csv("/Users/sylvain/Sofia/MSCS3807 Data Modeling in Python and R/imdb-data-modeling/imdb_top_1000.csv", stringsAsFactors = FALSE)

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

# Group sparse categorical levels to eliminate zero-variance folds
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
# 2. TRAIN-TEST SPLIT & 5-FOLD CV CONTROL
# -------------------------------------------------------------
set.seed(42)
train_index <- createDataPartition(model_data$IMDB_Rating, p = 0.80, list = FALSE)
train_set <- model_data[train_index, ]
test_set  <- model_data[-train_index, ]

ctrl <- trainControl(method = "cv", number = 5)

calc_metrics <- function(name, y_true, y_pred) {
  data.frame(
    Model = name,
    RMSE = round(RMSE(y_pred, y_true), 4),
    MAE = round(MAE(y_pred, y_true), 4),
    R2 = round(R2(y_pred, y_true), 4)
  )
}

# -------------------------------------------------------------
# 3. WEEK 2 BASELINES
# -------------------------------------------------------------
# Linear Regression
set.seed(42)
lr_fit <- train(IMDB_Rating ~ ., data = train_set, method = "lm", preProcess = c("zv", "center", "scale"), trControl = ctrl)
res_lr <- calc_metrics("1. Linear Regression", test_set$IMDB_Rating, predict(lr_fit, test_set))

# Decision Tree
set.seed(42)
dt_fit <- train(IMDB_Rating ~ ., data = train_set, method = "rpart", tuneGrid = expand.grid(cp = seq(0.001, 0.03, 0.005)), trControl = ctrl)
res_dt <- calc_metrics("2. Decision Tree", test_set$IMDB_Rating, predict(dt_fit, test_set))

# -------------------------------------------------------------
# 4. MODEL 3: RANDOM FOREST (caret / rf)
# -------------------------------------------------------------
set.seed(42)
# mtry: number of variables randomly sampled at each split
rf_grid <- expand.grid(mtry = c(2, 3, 4, 5))

rf_fit <- train(
  IMDB_Rating ~ .,
  data = train_set,
  method = "rf",
  tuneGrid = rf_grid,
  ntree = 200,
  importance = TRUE,
  trControl = ctrl
)
res_rf <- calc_metrics("3. Random Forest", test_set$IMDB_Rating, predict(rf_fit, test_set))

# -------------------------------------------------------------
# 5. MODEL 4: SUPPORT VECTOR REGRESSION (caret / svmRadial)
# -------------------------------------------------------------
set.seed(42)
svr_grid <- expand.grid(
  sigma = c(0.01, 0.05, 0.1),
  C = c(0.5, 1, 2, 4)
)

svr_fit <- train(
  IMDB_Rating ~ .,
  data = train_set,
  method = "svmRadial",
  preProcess = c("center", "scale"),
  tuneGrid = svr_grid,
  trControl = ctrl
)
res_svr <- calc_metrics("4. Support Vector Regression", test_set$IMDB_Rating, predict(svr_fit, test_set))

# -------------------------------------------------------------
# 6. SUMMARY EVALUATION & VARIABLE IMPORTANCE PLOT
# -------------------------------------------------------------
overall_results <- rbind(res_lr, res_dt, res_rf, res_svr)
cat("\n=== Comprehensive Model Evaluation Matrix (R caret) ===\n")
print(overall_results)

# Variable importance plot
rf_imp <- varImp(rf_fit)$importance
rf_imp$Feature <- rownames(rf_imp)

p_imp <- ggplot(rf_imp, aes(x = reorder(Feature, Overall), y = Overall)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "Feature Importance Ranking (Random Forest)",
    x = "Predictor Variable",
    y = "Importance Score"
  )

ggsave("r_week3_rf_feature_importance.png", plot = p_imp, width = 8, height = 5, dpi = 300)
print(p_imp)