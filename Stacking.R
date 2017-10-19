library(mlbench)
library(dplyr)
library(randomForest)
options(warn=-1)
##################
# Kfold function:
##################
Kfold = function(data,k=3){
        data = data[sample(nrow(data)),]
        drow = round(nrow(data)/k)
        datlist = list()
        for(i in seq(k)){
                if(i<k){

                        tempdat = data[drow*(i-1)+1:drow*i,]
                        print(dim(tempdat))
                        print(paste('from:',as.character(drow*(i-1)+1)))
                        print(paste('to:',as.character(drow*i)))
                        print(drow*i)
                        print('______')
                }else{
                        tempdat = data[(drow*(i-1)+1):nrow(data),]
                        print(dim(tempdat))
                        print(drow*(i-1)+1)
                        print(paste('to:',as.character(nrow(data))))
                        print('______')
        
                }
                datlist[[i]] = tempdat
        }
        return(datlist)
}


##################
# Load Data:
##################
data(BreastCancer)
dat = BreastCancer
#######################
# Basic Transfomation
#######################
dat = dat %>% dplyr::select(-Id)
dat = dat[complete.cases(dat),]
dat$Class =   as.numeric(dat$Class)
dat = as.data.frame(apply(dat,2,function(x){  as.numeric(x)   }))
dat[dat$Class==1,'Class'] = 0
dat[dat$Class==2,'Class'] = 1
#######################
# Train Valid:
#######################
set.seed(123)
dat = dat[sample(nrow(dat)),]
vl_size = round(nrow(dat)*0.2)
vali_dat = dat[1:vl_size,]
vali_dat_label = vali_dat['Class']
vali_dat = dplyr::select(vali_dat,-Class)
train_dat = dat[(vl_size+1):nrow(dat),]
################
# Get Kfold:
k = 3
Kfold_data = Kfold(train_dat,k)
################
################
# copy over the data for modeling,
# we don't want them have Stacking features added:
################
tr_kf = Kfold_data
vali_kf = vali_dat

#################
#Initialize Feature
################
Kfold_data[[1]][c('feat1','feat2','feat3')] = 0
Kfold_data[[2]][c('feat1','feat2','feat3')] = 0
Kfold_data[[3]][c('feat1','feat2','feat3')] = 0
vali_dat[c('feat1','feat2','feat3')] = 0




# LAYER 1:

#  Feature 1: Logistic Regression
test_pred = list()
for (i in seq(k)){
        print(paste('Fold',as.character(i)))
        temp_valid = tr_kf[[i]]
        temp_train = tr_kf
        temp_train[[i]] = NULL
        temp_train = do.call("rbind", temp_train)
        # Model:
        model = glm(Class~., data=temp_train,family=binomial(link='logit'))
        temp_valid_label = temp_valid['Class']
        temp_valid = dplyr::select(temp_valid,-Class)
        pred0 = predict(model,temp_valid)
        pred = ifelse(pred0>0.5,1,0)
        accuracy = sum(pred==temp_valid_label)/length(pred)
        print(paste('CV acc:',as.character(accuracy)))
        # Stack feature:
        Kfold_data[[i]]['feat1'] = pred
        #Test data feature:
        t_pred = predict(model,vali_kf)
        t_pred = ifelse(t_pred>0.5,1,0)
        test_pred[[i]] = t_pred
}

# Mean of each model for the feature in valid_dat:
vali_dat['feat1'] = round(rowMeans(sapply(test_pred, unlist)))


#  Feature 2: Random Forest
test_pred = list()
for (i in seq(k)){
        print(paste('Fold',as.character(i)))
        temp_valid = tr_kf[[i]]
        temp_train = tr_kf
        temp_train[[i]] = NULL
        temp_train = do.call("rbind", temp_train)
        # Model:
        model  = randomForest(as.factor(Class)~., data=temp_train)
        temp_valid_label = temp_valid['Class']
        temp_valid = dplyr::select(temp_valid,-Class)
        pred = predict(model,temp_valid)
        
        accuracy = sum(as.numeric(as.character(pred))==temp_valid_label)/length(pred)
        print(paste('CV acc:',as.character(accuracy)))
        # Stack feature:
        Kfold_data[[i]]['feat2'] = as.numeric(as.character(pred))
        
        #Test data feature:
        t_pred = predict(model,vali_kf)
        test_pred[[i]] = as.numeric(as.character(t_pred))
        
}

vali_dat['feat2'] = rowMeans(sapply(test_pred, unlist))




#  Feature 3: Naive Bayes:
test_pred = list()
for (i in seq(k)){
        print(paste('Fold',as.character(i)))
        temp_valid = tr_kf[[i]]
        temp_train = tr_kf
        temp_train[[i]] = NULL
        temp_train = do.call("rbind", temp_train)
        # Model:
        model = naiveBayes(as.factor(Class)~., data=temp_train)
        temp_valid_label = temp_valid['Class']
        temp_valid = dplyr::select(temp_valid,-Class)
        pred0 = predict(model,temp_valid,type='raw')[,2]
        pred = ifelse(pred0>0.5,1,0)
        
        
        accuracy = sum(as.numeric(as.character(pred))==temp_valid_label)/length(pred)
        print(paste('CV acc:',as.character(accuracy)))
        # Stack feature:
        Kfold_data[[i]]['feat3'] = pred
        
        #Test data feature:
        
        
        t_pred = predict(model,vali_kf,type='raw')[,2]
        t_pred = ifelse(t_pred>0.5,1,0)
        test_pred[[i]] = as.numeric(as.character(t_pred))
}

vali_dat['feat3'] = rowMeans(sapply(test_pred, unlist))








# Layer2:

#Ensemble:  0.9635036out-of-sample
l2_train = do.call('rbind',Kfold_data)
l2_train = l2_train[c('Cell.size','Cell.shape','feat1','feat2','feat3','Class')]
l2_valid = vali_dat[c('Cell.size','Cell.shape','feat1','feat2','feat3')]
l2_label = vali_dat_label

model = glm(Class~., data=l2_train,family=binomial(link='logit'))
final_res = predict(model,l2_valid)
final_res = ifelse(final_res>0.5,1,0)
sum(l2_label==as.numeric(as.character(final_res)))/length(final_res)






#Compare with single Model: 0.9489051 Single
md = glm(Class~., data=train_dat,family=binomial(link='logit'))
res = predict(md,vali_dat)
res = ifelse(res>0.5,1,0)
sum(l2_label==as.numeric(as.character(res)))/length(res)







#Compare with single Model: 0.97 Single
md = naiveBayes(as.factor(Class)~., data=train_dat)
res = predict(md,vali_dat)
sum(l2_label==as.numeric(as.character(res)))/length(res)
