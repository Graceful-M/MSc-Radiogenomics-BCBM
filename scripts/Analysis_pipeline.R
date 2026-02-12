# ===============================================================
# MSc Thesis: Predictive Models for Breast Cancer Brain Metastasis: An Integration of Neuroimaging and Genomics Data for Better Clinical Practice
# Author: Graceful Mulenga
# Institution: Botswana International University of Science and Technology (BIUST)
# Year: 2026
# ===============================================================

# ===============================
# 0. REQUIRED LIBRARIES
# ===============================

required_packages <- c(
  "readxl", "caret", "glmnet", "randomForest", "xgboost",
  "e1071", "pROC", "corrplot", "ggplot2",
  "dplyr", "reshape2", "RColorBrewer"
)

lapply(required_packages, library, character.only = TRUE)

set.seed(123)

# ===============================
# 1. DATA LOADING & PREPROCESSING
# ===============================

cat("Loading data...\n")

# Use relative path for GitHub reproducibility
data <- read_excel("data/BCBM_dataset.xlsx")

cat("Dataset dimensions:", dim(data), "\n")
cat("Class distribution:\n")
print(table(data$Metastatic_Status))

# Convert outcome to binary
data$target <- ifelse(data$Metastatic_Status == "Tumor", 1, 0)

# Select radiomic features only
feature_cols <- setdiff(names(data),
                        c("Age", "FilenamePrefix",
                          "Segmentation_Name",
                          "Metastatic_Status",
                          "target"))

data_clean <- data[, c(feature_cols, "target")]

# Remove missing values (if any)
data_clean <- na.omit(data_clean)

# Remove highly correlated features
correlation_matrix <- cor(data_clean[, -ncol(data_clean)])
high_corr <- findCorrelation(correlation_matrix, cutoff = 0.95)

if(length(high_corr) > 0) {
  data_clean <- data_clean[, -high_corr]
  cat("Removed", length(high_corr), "highly correlated features\n")
}

cat("Final dataset dimensions:", dim(data_clean), "\n")

# ===============================
# 2. HANDLE CLASS IMBALANCE
# ===============================

minority <- data_clean[data_clean$target == 1, ]
majority <- data_clean[data_clean$target == 0, ]

set.seed(123)
majority_sample <- majority[sample(nrow(majority), nrow(minority)), ]

balanced_data <- rbind(minority, majority_sample)
balanced_data$target <- as.factor(balanced_data$target)

levels(balanced_data$target) <- c("No_Tumor", "Tumor")

cat("Balanced class distribution:\n")
print(table(balanced_data$target))

# ===============================
# 3. TRAIN-TEST SPLIT
# ===============================

set.seed(123)

train_index <- createDataPartition(
  balanced_data$target,
  p = 0.8,
  list = FALSE
)

train_data <- balanced_data[train_index, ]
test_data  <- balanced_data[-train_index, ]

cat("Training size:", nrow(train_data), "\n")
cat("Test size:", nrow(test_data), "\n")

# ===============================
# 4. CROSS-VALIDATION SETUP
# ===============================

cv_control <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 3,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)

# ===============================
# 5. MODEL TRAINING
# ===============================

cat("Training models...\n")

# 5.1 LASSO
lasso_model <- train(
  target ~ .,
  data = train_data,
  method = "glmnet",
  trControl = cv_control,
  tuneGrid = expand.grid(
    alpha = 1,
    lambda = seq(0.001, 0.1, by = 0.001)
  ),
  metric = "ROC"
)

# 5.2 SVM (with scaling)
svm_model <- train(
  target ~ .,
  data = train_data,
  method = "svmRadial",
  trControl = cv_control,
  tuneGrid = expand.grid(
    sigma = c(0.01, 0.1, 1),
    C = c(0.1, 1, 10)
  ),
  metric = "ROC",
  preProc = c("center", "scale")
)

# 5.3 Random Forest
rf_model <- train(
  target ~ .,
  data = train_data,
  method = "rf",
  trControl = cv_control,
  tuneLength = 5,
  metric = "ROC",
  ntree = 500
)

# 5.4 XGBoost
xgb_grid <- expand.grid(
  nrounds = c(50, 100),
  max_depth = c(3, 6),
  eta = c(0.01, 0.1),
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 1
)

xgb_model <- train(
  target ~ .,
  data = train_data,
  method = "xgbTree",
  trControl = cv_control,
  tuneGrid = xgb_grid,
  metric = "ROC"
)

models <- list(
  LASSO = lasso_model,
  SVM = svm_model,
  Random_Forest = rf_model,
  XGBoost = xgb_model
)

# ===============================
# 6. MODEL EVALUATION
# ===============================

calculate_metrics <- function(actual, predicted, prob) {
  
  cm <- confusionMatrix(predicted, actual, positive = "Tumor")
  roc_obj <- roc(actual, prob, levels = c("No_Tumor", "Tumor"))
  
  return(data.frame(
    Accuracy = cm$overall["Accuracy"],
    Precision = cm$byClass["Pos Pred Value"],
    Recall = cm$byClass["Sensitivity"],
    F1_Score = cm$byClass["F1"],
    Specificity = cm$byClass["Specificity"],
    AUC = auc(roc_obj)
  ))
}

results <- data.frame()

for(name in names(models)) {
  
  cat("Evaluating", name, "\n")
  
  preds <- predict(models[[name]], test_data)
  probs <- predict(models[[name]], test_data, type = "prob")[, "Tumor"]
  
  metrics <- calculate_metrics(test_data$target, preds, probs)
  metrics$Model <- name
  
  results <- rbind(results, metrics)
}

results <- results[, c("Model", "Accuracy", "Precision",
                       "Recall", "F1_Score",
                       "Specificity", "AUC")]

print(results)

write.csv(results, "results/bcbm_model_results.csv",
          row.names = FALSE)

# ===============================
# 7. ROC CURVE VISUALIZATION
# ===============================

roc_data <- data.frame()

for(name in names(models)) {
  
  probs <- predict(models[[name]], test_data, type = "prob")[, "Tumor"]
  roc_obj <- roc(test_data$target, probs)
  
  roc_df <- data.frame(
    FPR = 1 - roc_obj$specificities,
    TPR = roc_obj$sensitivities,
    Model = name
  )
  
  roc_data <- rbind(roc_data, roc_df)
}

ggplot(roc_data, aes(FPR, TPR, color = Model)) +
  geom_line(size = 1) +
  geom_abline(linetype = "dashed") +
  theme_minimal() +
  labs(title = "ROC Curve Comparison",
       x = "False Positive Rate",
       y = "True Positive Rate")

# ===============================
# 8. FEATURE IMPORTANCE
# ===============================

cat("\nFeature Importance (Top 20 per model)\n")

for(name in names(models)) {
  
  if(name %in% c("Random_Forest", "XGBoost", "SVM")) {
    
    imp <- varImp(models[[name]])
    imp_df <- imp$importance
    imp_df$Feature <- rownames(imp_df)
    
    imp_df <- imp_df[order(-imp_df$Overall), ]
    
    cat("\n", name, "\n")
    print(head(imp_df, 20))
  }
}

# ===============================
# 9. BEST MODEL SELECTION
# ===============================

best_model <- results[which.max(results$AUC), ]

cat("\n=== BEST MODEL ===\n")
print(best_model)

# ===============================
# 10. SESSION INFO
# ===============================

sessionInfo()
