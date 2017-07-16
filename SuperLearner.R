data(BreastCancer)
dat = BreastCancer
library(dplyr)
dat = dat %>% select(-Id)
dat = dat[complete.cases(dat),]
dat$Class =   as.numeric(dat$Class)
dat = as.data.frame(apply(dat,2,function(x){  as.numeric(x)   }))

# change cat:
dat[dat$Class==1,"Class"] = 0
dat[dat$Class==2,"Class"] = 1
vl_size = 0.2

#shuffling:
dat = dat[sample(nrow(dat)),]
vl_size = round(nrow(dat)*vl_size)

# Train&Vali Split:
vali_dat = dat[1:vl_size,]
train_dat = dat[(vl_size+1):nrow(dat),]

super_train = as.data.frame(matrix(0,nrow = nrow(train_dat),ncol = 3),colnames=c('learner1','learner2','learner3'))

super_test= as.data.frame(matrix(0,nrow = nrow(vali_dat),ncol = 3),colnames=c('learner1','learner2','learner3'))



# Logistic Regression Meta Learner: 97.08%
lr_md = glm(Class~., data=train_dat,family=binomial(link='logit'))

pred = predict(lr_md, vali_dat[, colnames(vali_dat)!="Class"],type='response')

pred = ifelse(pred>0.5,1,0)
names(pred) = NULL
label = vali_dat$Class

caret::confusionMatrix(pred,label)

## For super learner:
sl_1 = predict(lr_md, train_dat[, colnames(train_dat)!="Class"],type='response')
sl_1 = ifelse(sl_1>0.5,1,0)
names(sl_1) = NULL
super_train[,1] = sl_1
super_test[,1] = pred


#Random Forest Meta Learner: 97.81%
library(randomForest)
rf_dat = train_dat
rf_dat$Class = as.factor(rf_dat$Class)
rf_md = randomForest(Class~., data=rf_dat)

pred2 = predict(rf_md, vali_dat[, colnames(vali_dat)!="Class"])
names(pred2) = NULL
pred2 = as.numeric(as.character(pred2))

caret::confusionMatrix(pred2,label)
## For super learner:
sl_2 = predict(rf_md, train_dat[, colnames(train_dat)!="Class"])
names(sl_2) = NULL
sl_2 = as.numeric(as.character(sl_2))
super_train[,2] = sl_2
super_test[,2] = pred2



# Naive Bayes Meta Learner: 96.35%
library(e1071)

nb_md = naiveBayes(Class~., data=train_dat)

pred3 = predict(nb_md, vali_dat[, colnames(vali_dat)!="Class"],type='raw')
pred3 = apply(pred3,1,function(x){ which.max(x) })
pred3 = pred3-1
caret::confusionMatrix(pred3,label)

## For super learner:
sl_3 = predict(nb_md, train_dat[, colnames(train_dat)!="Class"],type='raw')
sl_3 = apply(sl_3,1,function(x){ which.max(x) })
sl_3 = sl_3-1


super_train[,3] = sl_3
super_test[,3] = pred3



########### NeuralNetwork as the Super Learner: 98.54%
colnames(super_train) = colnames(super_test) = c('Learner1','Learner2','Learner3')

super_train$label = train_dat$Class
superModel = nnet(label~.,data=super_train, size = 2, rang = 0.1,decay = 5e-4, maxit = 200)

final_res = predict(superModel,super_test)

final = ifelse(final_res>0.5,1,0)

caret::confusionMatrix(final,label)

