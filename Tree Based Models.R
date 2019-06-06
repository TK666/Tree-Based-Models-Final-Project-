setwd("C:\\Users\\Lenovo\\Desktop\\Final Project")
getwd()


library(tidyverse)
library(rpart)
library(rpart.plot)
library(rattle)
library(Metrics)
library(randomForest)
library(xgboost)

Data <- read.csv("Data.csv")
ClData <- select(Data, - c("X", "Path", "Dists"))
str(ClData)

# Set seed and create assignment
set.seed(1)
assign <- sample(1:3, size = nrow(ClData), prob = c(0.7, 0.15, 0.15), replace = TRUE)

# Create a train and tests from the original data frame 
Data_train <- ClData[assign == 1, ]    # subset ETA_JAM to training indices only
Data_valid <- ClData[assign == 2, ]  # subset ETA_JAM to validation set only
Data_test <- ClData[assign == 3, ]   # subset ETA_JAM to test indices only

# Establish a list of possible values for minsplit and maxdepth
minsplit <- c(14,15,16)
maxdepth <- c(9,10,11)
cp <- c(0,0.0001,0.001,0.01)

# Create a data frame containing all combinations 
hyper_gridDT <- expand.grid(minsplit = minsplit, maxdepth = maxdepth, cp = cp)
nrow(hyper_gridDT)
# Create an empty list to store models
DT_models <- list()

# Write a loop over the rows of hyper_grid to train the grid of models
for (i in 1:nrow(hyper_gridDT)) {

  # Train a model and store in the list
  DT_models[[i]] <- rpart(formula = ETA_JAM ~ ., 
                             data = Data_train, 
                             method = "anova",
                             minsplit = hyper_gridDT$minsplit[i],
                             maxdepth = hyper_gridDT$maxdepth[i],
                             cp = hyper_gridDT$cp[i])
                            
}

# Create an empty vector to store RMSE values
rmse_DT <- c()

# Write a loop over the models to compute validation RMSE
for (i in 1:nrow(hyper_gridDT)) {
  
  # Generate predictions on grade_valid 
  pred_DT <- predict(object = DT_models[[i]],
                  newdata = Data_valid)
  
  # Compute validation RMSE and add to the 
  rmse_DT[i] <- rmse(actual = Data_valid$ETA_JAM, 
                         predicted = pred_DT)
}

# Identify the model with smallest validation set RMSE
Best_DT <- DT_models[[which.min(rmse_DT)]]

# Print the model paramters of the best model
Best_DT$control
Best_DT$ordered

# Compute test set RMSE on best_model
pred_DT1 <- predict(object = Best_DT,
                newdata = Data_test)
rmse(actual = Data_test$ETA_JAM, 
     predicted = pred_DT1)
# Plot the tree model
#fancyRpartPlot(Best_DT,
               #main = "Desicion Tree - Best Model ",palettes = "YlGn", type=1)

#######################

# Train a Random Forest
set.seed(2)  # for reproducibility
# Establish a list of possible values for mtry, nodesize and sampsize
mtry <- c(5,6,7)
nodesize <- c(5,7,9,11)
ntree <- c(35)

# Create a data frame containing all combinations 
hyper_gridRF <- expand.grid(mtry = mtry, nodesize = nodesize, ntree = ntree)
nrow(hyper_gridRF)
# Create an empty vector to store RF_models
RF_models <- c()

# Write a loop over the rows of hyper_grid to train the grid of models
for (i in 1:nrow(hyper_gridRF)) {

# Train a Random Forest model
RF_models[[i]] <- randomForest(formula = ETA_JAM ~ ., 
                      data = Data_train,
                      mtry = hyper_gridRF$mtry[i],
                      nodesize = hyper_gridRF$nodesize[i],
                      ntree = hyper_gridRF$ntree[i])

}
# Create an empty vector to store RMSE values
rmse_RF <- c()

# Write a loop over the models to compute validation RMSE
for (i in 1:nrow(hyper_gridRF)) {
  
  # Generate predictions on grade_valid 
  pred_RF <- predict(object = RF_models[[i]],
                     newdata = Data_valid)
  
  # Compute validation RMSE and add to the 
  rmse_RF[i] <- rmse(actual = Data_valid$ETA_JAM, 
                     predicted = pred_RF)
}

# Identify the model with smallest validation set RMSE
Best_RF <- RF_models[[which.min(rmse_RF)]]
Best_RF



# Compute test set RMSE on best_model
pred_RF1 <- predict(object = Best_RF,
                   newdata = Data_test)
rmse(actual = Data_test$ETA_JAM, 
     predicted = pred_RF1)


#XGBOOST
set.seed(3)
# Establish a list of possible values for mtry, nodesize and sampsize
maxdepthXGB <- c(6,7,8)
eta <- c(0.08)
nthreads <- c(6,7)
nrounds <- c(33,34)
subsample <- c(0.5)
min_child_weight <- c(5,7)
earlystoppingrounds <- c(3)
# Create a data frame containing all combinations 
hyper_gridXGB <- expand.grid(earlystoppingrounds=earlystoppingrounds, maxdepth = maxdepthXGB, eta = eta, nthreads = nthreads, nrounds = nrounds,subsample=subsample,min_child_weight=min_child_weight)
nrow(hyper_gridXGB)
# Create an empty vector to store RF_models
XGB_models <- c()

# Write a loop over the rows of hyper_grid to train the grid of models
for (i in 1:nrow(hyper_gridXGB)) {
  
  # Train a Random Forest model
  XGB_models[[i]] <- xgboost(data = as.matrix(Data_train),
                             label = Data_train$ETA_JAM,
                             max.depth = hyper_gridXGB$maxdepth[i],
                             eta = hyper_gridXGB$eta[i],
                             nthread = hyper_gridXGB$nthreads[i],
                             nrounds = hyper_gridXGB$nrounds[i],
                             min_child_weight = hyper_gridXGB$min_child_weight[i],
                             subsample = hyper_gridXGB$subsample[i],
                             early_stopping_rounds = hyper_gridXGB$earlystoppingrounds[i],
                             objective = "reg:linear")
  
}
# Create an empty vector to store RMSE values
rmse_XGB <- c()

# Write a loop over the models to compute validation RMSE
for (i in 1:nrow(hyper_gridXGB)) {
  
  # Generate predictions on grade_valid 
  pred_XGB <- predict(object = XGB_models[[i]],
                     newdata = as.matrix(Data_valid))
  
  # Compute validation RMSE and add to the 
  rmse_XGB[i] <- rmse(actual = Data_valid$ETA_JAM, 
                     predicted = pred_XGB)
}

# Identify the model with smallest validation set RMSE
Best_XGB <- XGB_models[[which.min(rmse_XGB)]]
Best_XGB

# Compute test set RMSE on best_model
pred_XGB1 <- predict(object = Best_XGB,
                    newdata = as.matrix(Data_test))
rmse(actual = Data_test$ETA_JAM, 
     predicted = pred_XGB1)

bst <- xgb.cv(data = as.matrix(ClData),
              label = ClData$ETA_JAM, nfold = 10,
              nrounds = 35, objective = "reg:linear", maximize = T)
