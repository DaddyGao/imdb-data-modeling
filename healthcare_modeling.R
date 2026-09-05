library(dplyr)
library(caret)
library(ggplot2)
library(randomForest)
library(e1071)

# -------------------------------------------------------------
# 1. LOAD DATA
# -------------------------------------------------------------
df <- read.csv("/Users/sylvain/Sofia/MSCS3807 Data Modeling in Python and R/imdb-data-modeling/insurance.csv", stringsAsFactors = FALSE)
cat("Dataset Dimensions:", dim(df), "\n")

# Convert categoricals to factors
model_data <- df %>%
  mutate(
    sex = as.factor(sex),
    smoker = as.factor(smoker),
    region = as.factor(region)
  )

# -------------------------------------------------------------
# 2. TRAIN-TEST SPLIT & 5-FOLD CV CONTROL
# -------------------------------------------------------------
set.seed(42)
train_index <- createDataPartition(model_data$charges, p = 0.80, list = FALSE)
train_set <- model_data[train_index, ]
test_set  <- model_data[-train_index, ]

ctrl <- trainControl(method = "cv", number = 5)

calc_metrics <- function(name, y_true, y_pred) {
  data.frame(
    Model = name,
    RMSE = round(RMSE(y_pred, y_true), 2),
    MAE = round(MAE(y_pred, y_true), 2),
    R2 = round(R2(y_pred, y_true), 4)
  )
}

# -------------------------------------------------------------
# 3. MODEL 1: LINEAR REGRESSION (OLS)
# -------------------------------------------------------------
set.seed(42)
lr_fit <- train(
  charges ~ .,
  data = train_set,
  method = "lm",
  preProcess = c("center", "scale"),
  trControl = ctrl
)
res_lr <- calc_metrics("1. Linear Regression", test_set$charges, predict(lr_fit, test_set))

# -------------------------------------------------------------
# 4. MODEL 2: DECISION TREE (rpart TUNED)
# -------------------------------------------------------------
set.seed(42)
dt_grid <- expand.grid(cp = seq(0.001, 0.05, by = 0.005))
dt_fit <- train(
  charges ~ .,
  data = train_set,
  method = "rpart",
  tuneGrid = dt_grid,
  trControl = ctrl
)
res_dt <- calc_metrics("2. Decision Tree", test_set$charges, predict(dt_fit, test_set))

# -------------------------------------------------------------
# 5. MODEL 3: RANDOM FOREST (caret / rf)
# -------------------------------------------------------------
set.seed(42)
rf_grid <- expand.grid(mtry = c(2, 3, 4, 5))
rf_fit <- train(
  charges ~ .,
  data = train_set,
  method = "rf",
  tuneGrid = rf_grid,
  ntree = 200,
  importance = TRUE,
  trControl = ctrl
)
res_rf <- calc_metrics("3. Random Forest", test_set$charges, predict(rf_fit, test_set))

# -------------------------------------------------------------
# 6. MODEL 4: SUPPORT VECTOR REGRESSION (svmRadial)
# -------------------------------------------------------------
set.seed(42)
svr_grid <- expand.grid(
  sigma = c(0.01, 0.05, 0.1),
  C = c(1000, 5000, 10000)
)
svr_fit <- train(
  charges ~ .,
  data = train_set,
  method = "svmRadial",
  preProcess = c("center", "scale"),
  tuneGrid = svr_grid,
  trControl = ctrl
)
res_svr <- calc_metrics("4. Support Vector Regression", test_set$charges, predict(svr_fit, test_set))

# -------------------------------------------------------------
# 7. SUMMARY TABLE & VARIABLE IMPORTANCE PLOT
# -------------------------------------------------------------
overall_results <- rbind(res_lr, res_dt, res_rf, res_svr)
cat("\n=== Week 4 Healthcare Application Model Evaluation (R caret) ===\n")
print(overall_results)

# Feature importance plot
rf_imp <- varImp(rf_fit)$importance
rf_imp$Feature <- rownames(rf_imp)

p_imp <- ggplot(rf_imp, aes(x = reorder(Feature, Overall), y = Overall)) +
  geom_col(fill = "teal") +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "Healthcare Feature Importance Ranking (Random Forest)",
    x = "Predictor Variable",
    y = "Importance Score"
  )

ggsave("r_week4_healthcare_importance.png", plot = p_imp, width = 8, height = 5, dpi = 300)