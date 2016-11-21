train_data<-read.table("/Users/polyM/Downloads/elections_usa96_train.csv", sep=",", header=T)
#data predprocessing
train_data<-na.omit(train_data)
levels(train_data$ClinLR)<-c(1,5,3,0,6,2,4)
levels(train_data$DoleLR)<-c(1,5,3,0,6,2,4)
levels(train_data$educ)<-c(5,4,3,2,1,6,0)
levels(train_data$income)<-c(23,5,6,7,8,9,10,11,12,13,14,15,16,1,0,17,18,19,20,3,21,3,22,4)
train_data$ClinLR<-as.numeric(as.character(train_data$ClinLR))
train_data$DoleLR<-as.numeric(as.character(train_data$ClinLR))
train_data$educ<-as.numeric(as.character(train_data$ClinLR))
train_data$income<-as.numeric(as.character(train_data$ClinLR))

set.seed(123)#for reproducability
#setting train and test set
index <- sample(1:nrow(train_data),size = 0.7*nrow(train_data)) 
train<-train_data[index,]
test<-train_data[-index,]
par <- c(0.01,0.05,0.1,0.5,1,5,10,50,100,500,1000)
par <- expand.grid(par,par)
dimnames(par)[[2]] <- c("C","sigma")
model <- ksvm(vote~(popul+TVnews+ClinLR+DoleLR+age+educ+income), train, type="C-svc", C = 0.1, kern = "rbfdot",kpar = list(sigma=0.1))
library(kernlab)
#fitStats function to calculate accuracy, precision, recall and F1 score of predictions
fitStats<-function(y, y_pred) {
  cm = as.matrix(table(Actual = y, Predicted = y_pred))
  n = sum(cm) # number of instances
  nc = nrow(cm) # number of classes
  diag = diag(cm) # number of correctly classified instances per class 
  rowsums = apply(cm, 1, sum) # number of instances per class
  colsums = apply(cm, 2, sum) # number of predictions per class
  p = rowsums / n # distribution of instances over the actual classes
  q = colsums / n # distribution of instances over the predicted classes
  accuracy=sum(diag(cm))/sum(cm)
  precision = diag / colsums 
  f1_score <- (2*precision*recall)/(precision + recall)
  Precision = mean(precision)
  Recall = mean(recall)
  F1 = mean(f1_score)
  stat <- c(accuracy,Precision,Recall,F1)
  names(stat) <- c("accuracy","precision","recall","f1.score")
  stat
}
#hyperparameters search
res<-NULL
for (i in 1:nrow(par)) { 
  model <- ksvm(vote~(popul+TVnews+ClinLR+DoleLR+age+educ+income), train, type="C-svc", C = par$C[i], kern = "rbfdot",
                kpar = list(sigma=par$sigma[i]))
  y_pred <- predict(model, newdata = test[,1:7], type = "response")
  res <- rbind(res, c(par$C[i],par$sigma[i],fitStats(test[,8],y_pred)) )
}
dimnames(res)[[2]][1:2] <- c("C","sigma")
j <- which.max(res[,"f1.score"])
res[j,]#the best parameters maximizing F1 score
best_model <- ksvm(vote~(popul+TVnews+ClinLR+DoleLR+age+educ+income), train, type="C-svc", C = res[j,1], kern = "rbfdot",
              kpar = list(sigma=res[j,2]))#best model
test_data<-read.table("/Users/polyM/Downloads/elections_usa96_test.csv", sep=",", header=T)
#test data predprocessing
test_data<-na.omit(test_data)
levels(test_data$ClinLR)<-c(1,5,3,0,6,2,4)
levels(test_data$DoleLR)<-c(1,5,3,0,6,2,4)
levels(test_data$educ)<-c(5,4,3,2,1,6,0)
levels(test_data$income)<-c(23,5,6,7,8,9,10,11,12,13,14,15,16,1,0,17,18,19,20,3,21,3,22,4)
test_data$ClinLR<-as.numeric(as.character(test_data$ClinLR))
test_data$DoleLR<-as.numeric(as.character(test_data$ClinLR))
test_data$educ<-as.numeric(as.character(test_data$ClinLR))
test_data$income<-as.numeric(as.character(test_data$ClinLR))

predict<-predict(best_model,newdata=test_data)
write.table(predict, "/Users/polyM/Downloads/election_test_prediction.csv",sep=",", row.names=F)
