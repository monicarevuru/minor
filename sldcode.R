library('caret')
set.seed(1)
data<-read.csv("sld.csv", header=T, na.strings=c("","NA"))
str(data)
library(plyr)

#data$Reading = as.integer(data$Reading)
#data$Writing = as.integer(data$Writing)
#data$Copying = as.integer(data$Copying)
#data$Spelling = as.integer(data$Spelling)
#data$Maths = as.integer(data$Maths)
#data$Attention = as.integer(data$Attention)


data1 <- data.frame(lapply(data[,5:23], as.numeric),stringsAsFactors = FALSE)
data1$LD <- as.factor(data1$LD)
data1$LD <- factor(data1$LD,levels = c(1,2),labels = c("no", "yes"))

str(data1)
sum(is.na(data1))
data_processed <- data1

index <- createDataPartition(data_processed$LD, p=0.75, list=FALSE)

trainSet <- data_processed[index,]
testSet <- data_processed[-index,]
fitControl <- trainControl(method = "cv", number = 5, savePredictions = 'final', classProbs = TRUE)

outcomeName<-"LD"
#predictors<-c("Reading", "Writing", "Spelling", "Maths","Copying","Attention")

model_dt<-train(LD~., data=trainSet, method="glm",trControl=fitControl, tuneLength=7)
testSet$pred_dt<-predict(object = model_dt,testSet[,])
cm <- confusionMatrix(testSet$LD,testSet$pred_dt)
a<-cm$overall['Accuracy']
print(a)

model_knn<-train(LD~., data=trainSet, method="knn",trControl=fitControl, tuneLength=7)
testSet$pred_knn<-predict(object = model_knn,testSet[,])
cm <- confusionMatrix(testSet$LD,testSet$pred_knn)
b<-cm$overall['Accuracy']
print(b)

model_svm<-train(LD~., data=trainSet, method="svmRadial",trControl=fitControl, tuneLength=7)
testSet$pred_svm<-predict(object = model_svm,testSet[,])
cm <-confusionMatrix(testSet$LD,testSet$pred_svm)
c<-cm$overall['Accuracy']
print(c)

trainSet$OOF_pred_dt_yes<-model_dt$pred$yes[order(model_dt$pred$rowIndex)]
trainSet$OOF_pred_knn_yes<-model_knn$pred$yes[order(model_knn$pred$rowIndex)]
trainSet$OOF_pred_svm_yes<-model_svm$pred$yes[order(model_svm$pred$rowIndex)]
testSet$OOF_pred_dt_yes<-predict(model_dt,testSet[],type='prob')$yes
testSet$OOF_pred_knn_yes<-predict(model_knn,testSet[],type='prob')$yes
testSet$OOF_pred_svm_yes<-predict(model_svm,testSet[],type='prob')$yes

trainSet$OOF_pred_dt_no<-model_dt$pred$no[order(model_dt$pred$rowIndex)]
trainSet$OOF_pred_knn_no<-model_knn$pred$no[order(model_knn$pred$rowIndex)]
trainSet$OOF_pred_svm_no<-model_svm$pred$no[order(model_svm$pred$rowIndex)]
testSet$OOF_pred_dt_no<-predict(model_dt,testSet[],type='prob')$no
testSet$OOF_pred_knn_no<-predict(model_knn,testSet[],type='prob')$no
testSet$OOF_pred_svm_no<-predict(model_svm,testSet[],type='prob')$no

predictors_top<-c('OOF_pred_dt_yes','OOF_pred_knn_yes','OOF_pred_svm_yes','OOF_pred_dt_no','OOF_pred_knn_no','OOF_pred_svm_no')
model_logit<-train(trainSet[,predictors_top],trainSet[,outcomeName],method='rpart',trControl=fitControl,tuneLength=7)
testSet$stacked<-predict(model_logit,testSet[,predictors_top])
cm <- confusionMatrix(testSet$LD,testSet$stacked)
d <- cm$overall["Accuracy"]
print(d)


#plotting different models accuracy for comparision 
df<-c("Dt","knn","svm","comb")
df1<-c(a,b,c,d)
cols <- c("red","lightblue","green","yellow")
barplot(df1,names.arg = df,xlab="Algorithms",ylab="Accuracy",col=cols, ylim = c(0,1), main="Comparision of algorithms")


print(testSet$stacked, mode = testSet$stacked$mode, digits = max(3,getOption("digits")-3),printStats = TRUE)
