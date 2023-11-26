#Reading Data from CSV file
trans = read.csv("C:/Gokul/VIT/Sem-7/CSE3020-DV/Project/transactions.csv")

#Attributes / Columns
names(trans)

#Head of trans data frame
View(head(trans))

#No: of rows in trans data frame
cat("No: of records: ",nrow(trans))


#Frequency Table of Fraudulent and Safe Transactions
#If Class is 1 - Fraudulent Transaction
#If Class is 0 - Safe Transaction
classFreq=table(trans$Class)
classFreq

data=data.frame(Class = names(classFreq), Count = as.vector(classFreq))

#Heat Map
correlation_matrix <- cor(trans[, -c(1, 31)])  # Exclude 'Time' and 'Class' columns
library(ggplot2)

# Melt the correlation matrix to long format
library(reshape2)
melted_corr <- melt(correlation_matrix)

# Create the heatmap using ggplot2
ggplot(data = melted_corr, aes(Var1, Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient(low = "pink", high = "black") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Correlation Heatmap")


#Pie chart
library(ggplot2)
ggplot(data, aes(x = "", y = Count, fill = Class)) +
  geom_bar(stat = "identity", width = 1) +
  coord_polar(theta = "y") +
  theme_void() +
  labs(fill = "Class")+
  labs(title = "Distribution of Fraud and Non-Fraud transactions")

#Frequency Chart
plot(classFreq, xlab = 'Class', ylab = 'Count', col = c("blue", "red"))
title(main = "Frequency of Fraudulent and Non-Fraudulent Transactions")

#Plot of time vs. Number of transactions for fraudulent and non- fraudulent.
fraudulent_data = trans[trans$Class == 1, ]
non_fraudulent_data = trans[trans$Class == 0, ]

# Create a plot for fraudulent transactions
ggplot(fraudulent_data, aes(x = Time)) +
  geom_histogram(binwidth = 1000, fill = "red", alpha = 0.7) +
  labs(title = "Time vs. Number of Fraudulent Transactions",
       x = "Time",
       y = "Number of Fraudulent Transactions") +
  theme_minimal()

# Create a plot for non-fraudulent transactions
ggplot(non_fraudulent_data, aes(x = Time)) +
  geom_histogram(binwidth = 1000, fill = "blue", alpha = 0.7) +
  labs(title = "Time vs. Number of Non-Fraudulent Transactions",
       x = "Time",
       y = "Number of Non-Fraudulent Transactions") +
  theme_minimal()


#Summary
summary(trans$Amount)

#SD and Var of Amount
var(trans$Amount)
sd(trans$Amount)


# Scaling Amount
#scale() : re-scale a vector by centering and scaling its values to have 
#zero mean and unit variance.
trans$Amount=scale(trans$Amount)
View(head(trans))

#SD and Var of Amount after re-scaling
var(trans$Amount)
sd(trans$Amount)

mean(trans$Amount)

#Eliminating time attribute
NewData=trans[,-c(1)]
View(head(NewData))


#(I) Logistic Regression Model

# Data Modelling
library(caTools)
set.seed(123)
data_sample = sample.split(NewData$Class,SplitRatio=0.75)
train_data = subset(NewData,data_sample==TRUE)
test_data = subset(NewData,data_sample==FALSE)
dim(train_data)
dim(test_data)

Logistic_Model=glm(Class~.,test_data,family=binomial())

summary(Logistic_Model)

# ROC Curve to assess the performance of the model
library(pROC)

#Prediction
lr.predict = predict(Logistic_Model, test_data, type="response")
# Set threshold for predicted class labels
threshold = 0.5
auc.gbm = roc(test_data$Class, lr.predict, plot = TRUE, col = "blue")
# Convert predicted probabilities to predicted class labels
predicted_class = ifelse(lr.predict > threshold, 1, 0)
head(predicted_class)

# Compare predicted class labels to actual class labels
accuracy = mean(predicted_class == test_data$Class)

# Print accuracy
cat("Accuracy of logistic regression model:", accuracy)

#ROC Curve
lr_auc = roc(test_data$Class, lr.predict, plot = TRUE, col = "blue",main = "ROC Curve for Logistic Regression")

#(II) Decision Tree Model
library(rpart)
library(rpart.plot)

# Data Modelling
set.seed(123)
data_sample = sample.split(NewData$Class,SplitRatio=0.75)
train_data = subset(NewData,data_sample==TRUE)
test_data = subset(NewData,data_sample==FALSE)

# Build decision tree model on training data

decisionTree_model = rpart(Class ~ ., data = train_data, method = "class")

# Generate predictions on test data
predicted_labels = predict(decisionTree_model, newdata = test_data, type = "class")

# Convert predicted_labels and test_data$Class to factors with levels 0 and 1
predicted_labels = factor(predicted_labels, levels = c(0, 1))
test_data$Class = factor(test_data$Class, levels = c(0, 1))


# Compute confusion matrix and accuracy
library(caret)
conf_matrix = confusionMatrix(predicted_labels, test_data$Class)
conf_matrix
accuracy = conf_matrix$overall["Accuracy"]
cat("Accuracy of Decision Tree model:", accuracy)

# Plot decision tree model
rpart.plot(decisionTree_model)

#ROC Curve
dt_predict = predict(decisionTree_model, newdata = test_data, type = "prob")[,2]
dt_auc = roc(test_data$Class, dt_predict, plot = TRUE, col = "red",main = "ROC Curve for Decision Tree")

#(III) Artificial Neural Network
library(neuralnet)

# Data Modelling
set.seed(123)
data_sample = sample.split(NewData$Class,SplitRatio=0.75)
train_data = subset(NewData,data_sample==TRUE)
test_data = subset(NewData,data_sample==FALSE)

ANN_model =neuralnet (Class~.,train_data,linear.output=FALSE)

plot(ANN_model)
predANN=compute(ANN_model,test_data)
resultANN=predANN$net.result
resultANN=ifelse(resultANN>0.5,1,0)

# Convert resultANN and test_data$Class to factors with levels 0 and 1
resultANN = factor(resultANN, levels = c(0, 1))
test_data$Class = factor(test_data$Class, levels = c(0, 1))

# Compute the confusion matrix for the predicted labels
conf_matrix = confusionMatrix(resultANN, test_data$Class)

# Extract the accuracy from the confusion matrix
accuracy = conf_matrix$overall['Accuracy']

# Print the accuracy
cat("Accuracy of ANN Model:", accuracy)

# ROC Curve for ANN model
roc_ann = roc(test_data$Class, as.vector(resultANN), plot = TRUE, col = "green")


#(IV) Gradient Boosting (GBM) (Ensemble Technique)
library(gbm, quietly=TRUE)

# Data Modelling
set.seed(123)
data_sample = sample.split(NewData$Class,SplitRatio=0.75)
train_data = subset(NewData,data_sample==TRUE)
test_data = subset(NewData,data_sample==FALSE)

# Get the time to train the GBM model
system.time(
model_gbm <- gbm(Class ~ .
                 , distribution = "bernoulli"
                 , data = rbind(train_data, test_data)
                 , n.trees = 500
                 , interaction.depth = 3
                 , n.minobsinnode = 100
                 , shrinkage = 0.01
                 , bag.fraction = 0.5
                 , train.fraction = nrow(train_data) / (nrow(train_data) + nrow(test_data))
)
)

# Determine best iteration based on test data
gbm.iter = gbm.perf(model_gbm, method = "test")
model.influence = relative.influence(model_gbm, n.trees = gbm.iter, sort. = TRUE)

#Plot the gbm model
plot(model_gbm)

# Plot and calculate AUC on test data
library(pROC)
gbm_test = predict(model_gbm, newdata = test_data, n.trees = gbm.iter)
gbm_auc = roc(test_data$Class, gbm_test, plot = TRUE, col = "orange")
print(gbm_auc)


# Predict Class values using the gbm model
predicted_values = predict(model_gbm, newdata = test_data, n.trees = gbm.iter, type = "response")

# Convert predicted values to binary predictions based on 0.5 threshold
binary_predicted_values = ifelse(gbm_test > 0.5, 1, 0)

# Calculate the accuracy
accuracy = sum(binary_predicted_values == test_data$Class) / length(test_data$Class)

# Print the accuracy
cat("Accuracy of GBM: ",accuracy)


