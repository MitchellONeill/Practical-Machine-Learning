---
title: "Practical Machine Learning Course Project"
author: "Mitchell O'Neill"
date: "Friday, May 08, 2015"
output: html_document
---


```
## Warning: package 'caret' was built under R version 3.1.3
```

```
## Warning: package 'rattle' was built under R version 3.1.3
```

##Executive Summary

There is a long hisory of quantifying strength training results in terms of repetitions and weight, but much less has been done to quantitatively analyze the correctness of form. In this project I  anaylzed the Weight Lifting Excercise Dataset (http://groupware.les.inf.puc-rio.br/har) and created a model that predicts the 'correctness' of form using accelerometer data from 4 sensors placed on participants performing barbell lifts. The WLE dataset has an observed variable for form called "classe" which places a lift into 5 lettered categories with 'A' representing proper form and 'B:E' describing certain common errors. The results of our model show that we can better predict the category of the variable 'classe' with a random forest model than a k-means (knn) one. We can also maintain a high accuracy and low out of sample expected error when keeping only 26 of the variables as predictors after performing some dataset tidying and preprocessing with principal components analysis. 

##Exploratory Analysis:

###Downloading and reading the datasets into [R]:


```r
set.seed(2015) ##setting a seed across the whole assignment for reproducibility
if(!file.exists("pml-training.csv")){
trainURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
download.file(trainURL, "pml-training.csv")
}
if(!file.exists("pml-testing.csv")){
testURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(testURL, "pml-testing.csv")
}
train <- read.csv(file="pml-training.csv", header=T, na.strings=c("NA", "")) ##NA's are entered both explicitly and as blanks in the dataset
test <- read.csv(file="pml-testing.csv", header=T, na.strings=c("NA", ""))
```

###Examining the data:


```r
dim(train); sum(is.na(train))
```

```
## [1] 19622   160
```

```
## [1] 1921600
```

```r
dim(test); sum(is.na(test)) 
```

```
## [1]  20 160
```

```
## [1] 2000
```

Our Training dataset has 160 variables with 19,622 observations. However, there are many NA values. Removing columns with missing values will give us tidy datasets to begin our analysis with. We should also partition the training dataset so that we can perform some cross-validation before applying our model to the testing dataset.


```r
Ttrain <- train[,colSums(is.na(train))==0]
Ttest <- test[,colSums(is.na(test))==0]
Ttrain <- Ttrain[, -c(1:7)]##further, columns 1:7 are only identifiers not related to the accelerometer data.
Ttest <- Ttest[, -c(1:7)]
dim(Ttrain)
```

```
## [1] 19622    53
```

```r
dim(Ttest)
```

```
## [1] 20 53
```

```r
summary(Ttrain$classe)
```

```
##    A    B    C    D    E 
## 5580 3797 3422 3216 3607
```

The summary of the 'classe' variable in our training set shows that each form level is relatively popular, as such we do not have to be as aware of skewedness or perform any transformations such as logging the results to make them more readable. 


```r
foldsTrain <- createFolds(y=Ttrain$classe, k=2, list=T, returnTrain=F) ##Partitions our training data into 2 sets
subTrain <- Ttrain[foldsTrain[[1]],]
subValidate <- Ttrain[foldsTrain[[2]],]
```

##Model Creation


###Random Forest Model: 


```r
sum(nearZeroVar(subTrain)) ##preliminary examination to see if any of our 52 predictors have no variance and can be removed
```

```
## [1] 0
```

```r
preProc <- preProcess(subTrain[,-53], method="pca", thresh=0.95) ##PCA will trim correlated variables from the dataset as long as 95% of the variation is still captured
subTrainPC <- predict(preProc, subTrain[,-53])
subValidatePC <- predict(preProc, subValidate[,-53])
modelRF <- suppressWarnings(train(subTrain$classe~., method="rf", data=subTrainPC, preProcess=c("center", "scale")))
```

```
## Loading required package: randomForest
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
 ##"center" and "scale" will standardize the variable so that variables with high mean and variability do not skew the data.
conRF <- confusionMatrix(subValidate$classe, predict(modelRF, subValidatePC))
kable(conRF$table); print(round(conRF$overall[1], 3))
```



|   |    A|    B|    C|    D|    E|
|:--|----:|----:|----:|----:|----:|
|A  | 2761|   10|   13|    5|    1|
|B  |   60| 1787|   44|    6|    2|
|C  |    5|   37| 1648|   20|    1|
|D  |    4|   15|  102| 1484|    3|
|E  |    3|   15|   27|   17| 1741|

```
## Accuracy 
##     0.96
```

Our 1st model uses the 1st fold of the training set and the random forest method from caret. Testing this model against the validation fold shows that we can predict the 'classe' variable with around 96% accuracy. 

###KNN models:


```r
modelKNN <- train(subTrain$classe~., method ="knn", preProcess = c("center", "scale"), tuneLength=10, trControl = trainControl(method = "boot"), data=subTrainPC)
conKNN<-confusionMatrix(subValidate$classe, predict(modelKNN, subValidatePC))
kable(conKNN$table);print(round(conKNN$overall[1], 3))
```



|   |    A|    B|    C|    D|    E|
|:--|----:|----:|----:|----:|----:|
|A  | 2720|   21|   26|   21|    2|
|B  |  101| 1696|   85|   11|    6|
|C  |   14|   53| 1587|   48|    9|
|D  |    8|    7|  122| 1467|    4|
|E  |    7|   30|   49|   33| 1684|

```
## Accuracy 
##    0.933
```

For our KNN model, I have chosen a tuneLength of 10. This means that 10 different K values (number of clusters) were used to find the best fit. The best fit value was 5, and our model predicted the classe variable in the validation fold with __ accuracy.  

###Comparing Models


```r
KNNSenSpe <- conKNN$byClass[,c("Sensitivity", "Specificity")]
RFSenSpe <- conRF$byClass[,c("Sensitivity", "Specificity")]
SensitivityCompare <- rbind(KNNSenSpe, RFSenSpe)
kable(SensitivityCompare)
```



|         | Sensitivity| Specificity|
|:--------|-----------:|-----------:|
|Class: A |   0.9543860|   0.9899440|
|Class: B |   0.9385722|   0.9746377|
|Class: C |   0.8491172|   0.9843868|
|Class: D |   0.9284810|   0.9828696|
|Class: E |   0.9876833|   0.9853195|
|Class: A |   0.9745852|   0.9958441|
|Class: B |   0.9586910|   0.9859066|
|Class: C |   0.8985823|   0.9921023|
|Class: D |   0.9686684|   0.9850223|
|Class: E |   0.9959954|   0.9923106|

The RF model has a higher sensitivy measure when predicting any level of the classe variable for lifting form. In addition, The models have very similar abilities at predicting across the factor levels of classe. They are both best at predicting categorizing the form type E and least effective for type C. The high specificity for C from both models further proves that the error in the models is in detecting too few Cs. 


```r
modelcompare <- table(RF=predict(modelRF, subValidatePC), KNN=predict(modelKNN, subValidatePC))
kable(modelcompare)
```



|   |    A|    B|    C|    D|    E|
|:--|----:|----:|----:|----:|----:|
|A  | 2765|   26|   19|   20|    3|
|B  |   63| 1709|   69|   13|    9|
|C  |   12|   41| 1707|   59|   16|
|D  |    9|    4|   45| 1461|   14|
|E  |    5|   25|   27|   26| 1664|

###Out of Sample Error predictions

The model we will be using is the RandomForest version as it had better sensitivity and accuracy across all levels of the 'classe' variable. 

Using the Confusion Matrix of the RF model above and our cross validation accuracy, our expected out of sample error is 1 - the accuracy of the validation set or approximately 4%.

###Submitting for Testing

Now that the model has been chosen analysized, estimating its predicted accuracy and out of sample error we can submit it to the wesbite to see our results against the testing data


```r
testPC <- predict(preProc, Ttest[,-53])
answers = predict(modelRF, testPC)
answers
```

```
##  [1] B A A A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

After submitting online, 19 of the 20 answers produced by the RF tree method were found to be correct.
