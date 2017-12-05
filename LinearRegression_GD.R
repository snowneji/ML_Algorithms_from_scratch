data=read.table('ex1data2.txt',sep = ',')
X=data[,1:2]
Y=data[,3]

LR_G = function(X,Y,alpha,nrounds){
        ################################################################################
        ###############MultiVariate Linear Regression ##################################
        #############################Sep 28,2016########################################
        ###########################Author: Yifan Wang###################################
        ################################################################################
        X = as.matrix(X)
        ### Normalize X to allow gradient descent #####
        Xnormalize = function(X){
                n = ncol(X)
                mus = rep(0,n)
                sigmas = rep(0,n)
                
                for (i in 1:n){
                        mu = mean(X[,i])
                        sigma = sd(X[,i])
                        X[,i] = (X[,i] - mu)/sigma
                        mus[i] = mu
                        sigmas[i] = sigma
                }
                
                res = list(X,mus,sigmas)
                return(res)
        }
        nor_res = Xnormalize(X)
        X = nor_res[[1]]
        trans_mean = nor_res[[2]]
        trans_sd = nor_res[[3]]
        ##### add an intercept###
        X = cbind(1,X)
        ##### Initialize theta ######
        theta = as.matrix(rep(0,ncol(X)))
        
        ##### Cost Function #####
        J = function(X,Y,theta){
                m=nrow(X)
                theta=as.matrix(theta)
                J = (1/(2*m))*  t(X %*% theta - Y) %*% (X %*% theta - Y)
                J
                
        }
        
        
        
        #####Use Gradient Descent to Converge #######
        GraDes = function(X,Y,theta,alpha,nrounds){
                m = nrow(X)
                cost_hist = rep(0,nrounds)
                for (i in 1:nrounds){
                        cost = J(X,Y,theta)
                        print(paste('round:',i,'|||','cost:',cost))
                        cost_hist[i] = cost
        #  t(X %*% theta - Y) %*% X) is  the derivative of your cost function                        
                        theta = theta - t((alpha/m)* t(X %*% theta - Y) %*% X)
                        
                }
                res = list(theta,cost_hist)
                return(res)
                
        }
        
        md_res = GraDes(X,Y,theta,alpha,nrounds)
        
        res = list(md_res[[1]],md_res[[2]],trans_mean,trans_sd)
        names(res) = c('theta','cost_history','transformation_mean','transformation_sd')
        return(res)
        
}




LR_G_Predict = function(model,new_data){
        new_data = as.matrix(new_data)
        new_data = (new_data - model$transformation_mean)/model$transformation_sd
        new_data = cbind(1,new_data)
        
      
        
        result  = new_data %*% model$theta
        return(result)
}

