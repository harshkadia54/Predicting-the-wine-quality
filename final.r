wine <- read.csv("H:/MASTERS/Data Mining and Machine Learning/Assessment 2018-19/CW2/winequality-red.csv")

str(wine)      #Structure of dataset

dim(wine)       #Checking dimension of dataset

summary(wine)        #Checking summary of the dataset

apply(wine,2,function(x) length(unique(x)))       #Checking the number of unique values

table(wine$quality)

wine$quality <- as.integer(wine$quality)
par(mfrow = c(4,3))
for (i in c(1:11)) {
  plot(wine[, i], jitter(wine[, "quality"]), xlab = names(wine)[i],
       ylab = "quality", col = "firebrick", cex = 0.8, cex.lab = 1.3)
  abline(lm(wine[, "quality"] ~ wine[ ,i]), lty = 2, lwd = 2)
}
par(mfrow = c(1, 1))


par(mfrow = c(1,1))
cor.wine <- cor(wine)
corrplot(cor.wine, method = 'number')




# ## GGPlot
# library(ggplot2)
# pairs(wine[, 1:12])
# 
# ## GGally
# library(GGally)
# ggpairs(wine[, 1:11])
# 
# ## corrplot
# library(corrplot)
# wine_cor <- round(cor(wine[, 2:12]), 3)
# corrplot.mixed(wine_cor, lower = "number", upper = "color", tl.cex = 0.5)

## From the correlation plots we can see that free.sulfur.dioxide and 
## total.sulfur.dioxide are strongly correlated. Density and sulfates 
## are also strongly correlated. It is also observed that Quality and 
## alcohol are strongly associated.

# Remove Redundant Features
# ensure the results are repeatable
set.seed(123)
# load the library
library(mlbench)
library(caret)
# load the data
x <- wine
# calculate correlation matrix
correlationMatrix <- cor(x[,1:11])
# summarize the correlation matrix
print(correlationMatrix)
# find attributes that are highly corrected (ideally >0.75)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.5)
# print indexes of highly correlated attributes
print(highlyCorrelated)

#Rank Features By Importance
# ensure results are repeatable
set.seed(7)
# load the library
# library(mlbench)
# library(caret)
# load the dataset
x1<-wine[,-12]
quality <- wine$quality
# prepare training scheme
control <- trainControl(method="repeatedcv", number=10, repeats=3)
# train the model
model <- train(quality~., data=wine, method = "treebag",preProcess="scale")
# estimate variable importance
importance <- varImp(model, scale=FALSE)
# summarize importance
print(importance)
# plot importance
plot(importance)

# Feature Selection
set.seed(1234)
# load the data
x2 <- wine
# define the control using a random forest selection function
control <- rfeControl(functions=rfFuncs, method="cv", number=10)
# run the RFE algorithm
results <- rfe(wine[,1:11], wine[,12], sizes=c(1:11), rfeControl=control)
# summarize the results
print(results)
# list the chosen features
predictors(results)
# plot the results
plot(results, type=c("g", "o"))


####k-fold cross validation
# load the iris dataset
x3 <- wine
# define training control
train_control <- trainControl(method="cv", number=10)
# fix the parameters of the algorithm
grid <- expand.grid(.fL=c(0), .usekernel=c(FALSE))
# train the model
ControlParameters <- trainControl(method = "cv", number = 5, savePredictions = TRUE, classProbs = TRUE)
#trainControl method tells which method you are trying 
#to use. CV means cross validation. number = 5 means 5 fold cross validation i.e. we will create 5 partition of the 
#dataset. savePredictions will save the prediction of the model. class Probabilities will see the probabilities predicted 
#by the model.

# summarize results
print(model)



####RANDOM FOREST################# 

head(wine)
wine$quality = as.factor(wine$quality)

wine_set_size= floor(nrow(wine)*0.80)
index <- sample(1:nrow(wine), size = wine_set_size)

training <- wine[index,]
testing <- wine[-index,]

library(randomForest)             #Used for both classification and regression
rf <- randomForest(quality ~ ., data = training, mtry=4, ntree=2001, importance=TRUE)
rf                        #The error rate is 29.24%. Therefore, accuracy is 70.76%. Always check your 
#domain when you are sharing your stuff because if you are doing something that is life or a death maybe 70% accuracy 
#might not be so good but if you are looking at where you are going to distribute the bottle's to for the wine
#and who is going to drink it, 70% accuracy is pretty good for that.

plot(rf) #As the number of forest grows the error rates are displayed in the line graph

result <- data.frame(testing$quality, predict(rf,testing[,1:11], type = "response"))   #Predict
result

plot(rf)
plot(result)

library(ROCR)
pred <- predict(ControlParameters, type = 'prob')  

confusionMatrix(result, testing$quality)        ####ERROR
misclassificationerror <- mean(pred != wine$quality)

####NAIVE BAYES
library(klaR)
# define an 80%/20% train/test split of the dataset
split=0.80
trainIndex <- createDataPartition(wine$quality, p=split, list=FALSE)
data_train <- wine[ trainIndex,]
data_test <- wine[-trainIndex,]
# train a naive bayes model
model= naiveBayes(quality~., data = x)
model
# make predictions
x_train <- data_train[,1:11]
y_test <- data_test[,12]
predictions <- predict(model, x_train)

# summarize results
confusionMatrix(as.number(wine), as.number(testing$quality))




