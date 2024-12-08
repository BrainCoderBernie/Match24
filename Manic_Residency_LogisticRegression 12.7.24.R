# Set a seed for reproducibility
set.seed(42)

# Simulating data for 161 matched and 71 unmatched individuals
n_matched <- 161
n_unmatched <- 71
n_total <- n_matched + n_unmatched

# Simulate STEP2 scores
step2_matched <- rnorm(n_matched, mean = 255, sd = 12 / 1.349)  # IQR to std: IQR = 1.349 * std
step2_unmatched <- rnorm(n_unmatched, mean = 248, sd = 17 / 1.349)

# Simulate RESEARCH_TOT
research_tot_matched <- rnorm(n_matched, mean = 37.4, sd = 5)
research_tot_unmatched <- rnorm(n_unmatched, mean = 31.8, sd = 5)

# Simulate TOP40 (binary variable)
top40_matched <- rbinom(n_matched, 1, prob = 0.46)
top40_unmatched <- rbinom(n_unmatched, 1, prob = 0.25)

# Combine data into a single dataset
step2 <- c(step2_matched, step2_unmatched)
research_tot <- c(research_tot_matched, research_tot_unmatched)
top40 <- c(top40_matched, top40_unmatched)
matched <- c(rep(1, n_matched), rep(0, n_unmatched))  # 1 = matched, 0 = unmatched

# Create a DataFrame
data <- data.frame(
  STEP2 = step2,
  RESEARCH_TOT = research_tot,
  TOP40 = top40,
  MATCHED = matched
)

# Split data into training and testing sets (manually)
set.seed(42)
train_indices <- sample(1:nrow(data), size = 0.75 * nrow(data))
train_data <- data[train_indices, ]
test_data <- data[-train_indices, ]

# Logistic regression model
logistic_model <- glm(MATCHED ~ STEP2 + RESEARCH_TOT + TOP40, data = train_data, family = binomial)
summary(logistic_model)
exp(coef(logistic_model))

# Make predictions
test_data$predicted_prob <- predict(logistic_model, newdata = test_data, type = "response")
test_data$predicted_class <- ifelse(test_data$predicted_prob > 0.5, 1, 0)

# Evaluate the model
confusion_matrix <- table(test_data$MATCHED, test_data$predicted_class)
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)

# Calculate ROC and AUC
install.packages(pROC)
library(pROC)
roc_curve <- roc(test_data$MATCHED, test_data$predicted_prob)
auc_score <- auc(roc_curve)

# Print results
cat("Confusion Matrix:\n")
print(confusion_matrix)
cat("\nAccuracy:", accuracy, "\n")
cat("AUC:", auc_score, "\n")

# Plot the ROC curve
plot(roc_curve, main = "ROC Curve", col = "blue", lwd = 2)

# Define the individual's data
individual <- data.frame(
  STEP2 = 251,
  RESEARCH_TOT = 40,
  TOP40 = 1
)

# Predict the probability of matching
individual$predicted_prob <- predict(logistic_model, newdata = individual, type = "response")

# Output the probability
cat("Predicted Probability of Matching:", individual$predicted_prob, "\n")