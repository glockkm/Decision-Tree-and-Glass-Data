#Kimberly Glock
#11/06/2019
#MSDS 5213 Lab 2 Decision Trees

#http://archive.ics.uci.edu/ml/datasets/glass+identification
#Attribute Information:
#1. Id number: 1 to 214 
#2. RI: refractive index 
#3. Na: Sodium (unit measurement: weight percent in corresponding oxide, as are attributes 4-10) 
#4. Mg: Magnesium 
#5. Al: Aluminum 
#6. Si: Silicon 
#7. K: Potassium 
#8. Ca: Calcium 
#9. Ba: Barium 
#10. Fe: Iron 
#11. Type of glass: (class attribute) 


rm(list=ls())

library(rpart)
library(rattle)
library(caret)
library(caTools)
library(ROCR)
library(rpart.plot)
library(ipred)
library(randomForest)
library(dplyr)
library(forcats)

glass_with_id = read.csv("glass.csv", head=FALSE, sep = ",")
View(glass_with_id)
glass_with_id$V11 = factor(glass_with_id$V11, levels = c(1,2,3,5,6,7), 
                           labels=c("building_windows_float_processed",
                                    "building_windows_non_float_processed",
                                    "vehicle_windows_float_processed",
                                    "containers",
                                    "tableware",
                                    "headlamps"))
#Takes TARGET column/varaible and changes to a factor and then applies names
glass = glass_with_id[ -c(1) ] #take out id column
View(glass)
dim(glass)
summary(glass)

                        
set.seed(123)
split = sample.split(glass$V11, SplitRatio = 0.8)                              
train = subset(glass, split==TRUE)
test = subset(glass, split==FALSE)
test_no_class = subset(glass, select=-V11, split==FALSE)

fitcontrol = trainControl(method = "cv", number = 10) #set up cross validation
grid_cp = expand.grid(cp=seq(0, 0.05, 0.005)) #find best complexity parameter

mod = train(V11 ~ ., data=train, method = "rpart",  #model using caret
            trControl=fitcontrol, metric="Accuracy", 
            maximize=TRUE, tuneGrid=grid_cp)
mod


pred_reg = predict(mod, test_no_class, type="raw") #predict using model and test data without class

#confusionMatrix(factor(pred_reg), factor(glass$V11))
class = test$V11
confusionMatrix(pred_reg, class)

#best dec tree using cp=0.02
#best_cp_tree = rpart(V11~ ., data=train, cp=0.02)
#pred_cp_best = predict(best_cp_tree, test_no_class) 
#confusionMatrix(pred_cp_best, test$V11)

control=rpart.control(xval=10, minbucket=2, minsplit=4, cp=0.02)
control=rpart.control(cp=0.02)
mod_rp = rpart(V11~ ., data=train, control=control) #rpart model
fancyRpartPlot(mod_rp) #plot best decision tree

#test and cm on best model using cp = 0.02
pred_dec_best = predict(mod_rp, test_no_class) 
###### help       confusionMatrix(pred_dec_best, class)
## Test the best cp model and create a confusion matrix

#```{r }
#pred_reg_best = predict(mod_rp, test_no_class) 
#confusionMatrix(pred_reg_best, class)
#```


####### BAGGING #################################################

split = sample.split(glass$V11, SplitRatio = 0.8)                              
train = subset(glass, split==TRUE)
test = subset(glass, split==FALSE)
test_no_class = subset(glass, select=-V11, split==FALSE)

#use for loop to determine best nbagg number
accu = rep(0,70)
for (i in 1:70) {
  set.seed(i)
  bagged = bagging(V11~ ., data=train, nbagg=i)
  out = predict(bagged, test_no_class)
  cm = confusionMatrix(out, test$V11)
  
  accu[i] = cm[3]$overall[1]
}

accu
max(accu) #50 nbagg is best
#set.seed(i) get same results for accuracy when running

#Create best model using nbagg = 50
bagged50 = bagging(V11~ ., data=train, nbagg=50) #bagging model

#produce best model in caret using nbagg = 50
mod_rp_caret50 = train(V11~ ., data=train, method="treebag",
                     trControl=fitcontrol, metric="Accuracy", nbagg=50, maximize=TRUE)
print(mod_rp_caret50)


pred_bag50 = predict(mod_rp_caret50, test_no_class)

#####class = glass$V11
#confusionMatrix(factor(pred_bag), factor(glass$V11))
#######confusionMatrix(pred_bag50, class)
confusionMatrix(pred_bag50, class)

####### RANDOM FOREST TREE ###########################################

#https://stackoverflow.com/questions/33038310/how-to-find-the-best-ntree-and-nodesize-in-randomforest-in-r-and-then-calculate

rf_model = randomForest(V11~ ., data=train)
rf_model

pred_rf = predict(rf_model, test_no_class)
confusionMatrix(pred_rf, class)

#use caret package to create a rf model
control_rand_for = trainControl(method="repeatedcv", number=10, repeats=3)
metric = "Accuracy"
n= round(sqrt(ncol(train)))
tunegrid = expand.grid(.mtry=seq(1:10))
rand_for = train(V11 ~ ., data=train, method="rf",
                 metric=metric, tuneGrid= tunegrid, trControl=control_rand_for)
print(rand_for)

pred_rf_caret = predict(rand_for, test_no_class)
######confusionMatrix(pred_rf_caret, factor(class))
confusionMatrix(pred_rf_caret, class)

important = varImp(rand_for) #sgows important variables
important


#for loop to determin best number of trees using ntree
accu = rep(0,1000)
for (i in 1:1000) {
  set.seed(i)
  rf_model = randomForest(V11~ ., data=train, ntree=i)
  pred_loop = predict(rf_model, test_no_class)
  cm = confusionMatrix(pred_loop, class)
  
  accu[i] = cm[3]$overall[1]
}

accu
max(accu)
#47 trees has a best accuracy of 0.7674419


#using 47 trees and now determining best mtry
control_rand_for47 = trainControl(method="repeatedcv", number=10, repeats=3)
metric = "Accuracy"
n= round(sqrt(ncol(train)))
tunegrid = expand.grid(.mtry=seq(n+1:10))
rand_for47 = train(V11 ~ ., data=train, method="rf", ntree=47,
                 metric=metric, tuneGrid= tunegrid, trControl=control_rand_for47)
print(rand_for47)
#mtry = 2 is best with 47 trees

#best model created and tested
rf_model_best = randomForest(V11~ ., data=train, ntree=47, mtry=2)
rf_model_best

pred_rf_best = predict(rf_model_best, test_no_class)
confusionMatrix(pred_rf_best, class)


#https://www.rdocumentation.org/packages/ROCR/versions/1.0-7/topics/performance
#https://www.rdocumentation.org/packages/rpart/versions/4.1-15/topics/rpart
#https://www.rdocumentation.org/packages/base/versions/3.6.1/topics/expand.grid
#datamining.togaware.com/survivor/Complexity_cp.html
#https://www.rdocumentation.org/packages/ipred/versions/0.9-9/topics/bagging
#https://www.rdocumentation.org/packages/caret/versions/6.0-84/topics/varImp
