#1. Exploratory Data Analysis (EDA)

# Load necessary libraries
library(tidyverse)
library(ggplot2)
library(caret)
library(corrplot)
library(reshape2)
library(pROC)

# Sampling data 
data <- read.csv("Data/heart_attack_prediction_dataset.csv")

# Initial exploration
str(data)  # View structure of the dataset
summary(data)  # Summary statistics
head(data)  # View the first few rows



# Handling Missing Values and Data Types
# Checking for missing values
if (sum(is.na(data)) > 0) {
  data <- na.omit(data)  # Remove missing values if they exist
}

# Converting categorical variables to factors
data$Sex <- as.factor(data$Sex)
data$Diabetes <- as.factor(data$Diabetes)
data$Family.History <- as.factor(data$Family.History)
data$Smoking <- as.factor(data$Smoking)
data$Obesity <- as.factor(data$Obesity)
data$Alcohol.Consumption <- as.factor(data$Alcohol.Consumption)
data$Previous.Heart.Problems <- as.factor(data$Previous.Heart.Problems)
data$Medication.Use <- as.factor(data$Medication.Use)
data$Country <- as.factor(data$Country)
data$Continent <- as.factor(data$Continent)
data$Hemisphere <- as.factor(data$Hemisphere)
data$Heart.Attack.Risk <- as.factor(data$Heart.Attack.Risk)

# View updated structure
str(data)

#2.Visualization
# Distribution of Heart Attack Risk
ggplot(data, aes(x = Heart.Attack.Risk)) +
  geom_bar(fill = "skyblue") +
  theme_minimal() +
  ggtitle("Distribution of Heart Attack Risk")

# Age distribution by Heart Attack Risk
ggplot(data, aes(x = Age, fill = Heart.Attack.Risk)) +
  geom_histogram(binwidth = 5, position = "dodge") +
  theme_minimal() +
  ggtitle("Age Distribution by Heart Attack Risk")

# Correlation matrix for numeric variables
numeric_vars <- select_if(data, is.numeric)
cor_matrix <- cor(numeric_vars)
corrplot(cor_matrix, method = "color", type = "upper", tl.cex = 0.8)


#3. Preprocessing
# Scaling numeric features
preprocessParams <- preProcess(numeric_vars, method = c("center", "scale"))
scaled_data <- predict(preprocessParams, numeric_vars)

# Combine with categorical data
final_data <- cbind(scaled_data, select_if(data, is.factor))

# Create training and test sets
set.seed(123)
trainIndex <- createDataPartition(final_data$Heart.Attack.Risk, p = 0.8, list = FALSE)
train_data <- final_data[trainIndex,]
test_data <- final_data[-trainIndex,]

#4.Modeling
#4.1 Logistic Regression
log_model <- glm(Heart.Attack.Risk ~ ., data = train_data, family = binomial)
summary(log_model)

# Predictions
log_predictions <- predict(log_model, test_data, type = "response")
log_predictions <- ifelse(log_predictions > 0.5, 1, 0)

#4.2 Random Forest
library(randomForest)
rf_model <- randomForest(Heart.Attack.Risk ~ ., data = train_data)
print(rf_model)

# Predictions
rf_predictions <- predict(rf_model, test_data)

#4.3 Support Vector Machines (SVM)
library(e1071)
svm_model <- svm(Heart.Attack.Risk ~ ., data = train_data, probability = TRUE)
summary(svm_model)

# Predictions
svm_predictions <- predict(svm_model, test_data)

#5.Evaluation

# Confusion Matrix and Accuracy for Logistic Regression
log_confusion <- confusionMatrix(as.factor(log_predictions), test_data$Heart.Attack.Risk)
log_confusion

# Confusion Matrix and Accuracy for Random Forest
rf_confusion <- confusionMatrix(rf_predictions, test_data$Heart.Attack.Risk)
rf_confusion

# Confusion Matrix and Accuracy for SVM
svm_confusion <- confusionMatrix(svm_predictions, test_data$Heart.Attack.Risk)
svm_confusion


# Function to plot confusion matrix
plot_confusion_matrix <- function(cm, title) {
  cm_table <- as.data.frame(cm$table)
  ggplot(data = cm_table, aes(x = Reference, y = Prediction, fill = Freq)) +
    geom_tile() +
    geom_text(aes(label = Freq), vjust = 1) +
    scale_fill_gradient(low = "white", high = "steelblue") +
    theme_minimal() +
    labs(title = title, x = "Actual", y = "Predicted")
}

# Confusion Matrix for Logistic Regression
log_confusion <- confusionMatrix(as.factor(log_predictions), test_data$Heart.Attack.Risk)
log_cm_plot <- plot_confusion_matrix(log_confusion, "Logistic Regression Confusion Matrix")

# Confusion Matrix for Random Forest
rf_confusion <- confusionMatrix(rf_predictions, test_data$Heart.Attack.Risk)
rf_cm_plot <- plot_confusion_matrix(rf_confusion, "Random Forest Confusion Matrix")

# Confusion Matrix for SVM
svm_confusion <- confusionMatrix(svm_predictions, test_data$Heart.Attack.Risk)
svm_cm_plot <- plot_confusion_matrix(svm_confusion, "SVM Confusion Matrix")

# Display the plots
print(log_cm_plot)
print(rf_cm_plot)
print(svm_cm_plot)

# Function to calculate and return evaluation metrics
evaluate_model <- function(confusion) {
  accuracy <- confusion$overall['Accuracy']
  precision <- confusion$byClass['Pos Pred Value']
  recall <- confusion$byClass['Sensitivity']
  f1 <- 2 * ((precision * recall) / (precision + recall))
  
  metrics <- c(accuracy, precision, recall, f1)
  names(metrics) <- c("Accuracy", "Precision", "Recall", "F1 Score")
  return(metrics)
}

# Evaluate Logistic Regression
log_metrics <- evaluate_model(log_confusion)

# Evaluate Random Forest
rf_metrics <- evaluate_model(rf_confusion)

# Evaluate SVM
svm_metrics <- evaluate_model(svm_confusion)

# Combine metrics into a single data frame
evaluation_results <- data.frame(
  Metric = c("Accuracy", "Precision", "Recall", "F1 Score"),
  Logistic_Regression = log_metrics,
  Random_Forest = rf_metrics,
  SVM = svm_metrics
)

print(evaluation_results)


# Calculate AUC for Logistic Regression
log_roc <- roc(test_data$Heart.Attack.Risk, as.numeric(log_predictions))
log_auc <- auc(log_roc)

# Calculate AUC for Random Forest
rf_roc <- roc(test_data$Heart.Attack.Risk, as.numeric(rf_predictions))
rf_auc <- auc(rf_roc)

# Calculate AUC for SVM
svm_roc <- roc(test_data$Heart.Attack.Risk, as.numeric(svm_predictions))
svm_auc <- auc(svm_roc)

# Add AUC to the evaluation results
evaluation_results <- rbind(evaluation_results, 
                            AUC = c(Logistic_Regression = log_auc, 
                                    Random_Forest = rf_auc, 
                                    SVM = svm_auc))

print(evaluation_results)


# Save the pre-processing parameters and the Random Forest model
saveRDS(preprocessParams, "preprocess_params.rds")
saveRDS(rf_model, "random_forest_model.rds")

