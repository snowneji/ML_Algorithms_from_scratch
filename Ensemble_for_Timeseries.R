#Reference: https://petolau.github.io/Ensemble-of-trees-for-forecasting-time-series/
library(feather) # data import
library(data.table) # data handle
library(rpart) # decision tree method
library(party) # decision tree method
library(forecast) # forecasting methods
library(randomForest) # ensemble learning method
library(ggplot2) # visualizations
library(gridExtra)
#LoadData:
DT <- as.data.table(read_feather("DT_load_17weeks"))
# DT <- as.data.frame(read_feather("DT_load_17weeks"))

#Info:
n_date <- unique(DT[, date]) # Unique dates
period <- 48 # 48 records per day

# Train&Test Split:
data_train <- DT[date %in% n_date[1:21]]
data_test <- DT[date %in% n_date[22]]


# Train Visualize:
trp = ggplot(data_train, aes(date_time, value)) +
  geom_line() +
  labs(x = "Date", y = "Load (kW)") +
  theme_ts

tsp = ggplot(data_test, aes(date_time, value)) +
  geom_line() +
  labs(x = "Date", y = "Load (kW)") +
  theme_ts

grid.arrange(trp,tsp,nrow=2)



#Modeling
data_ts <- ts(data_train$value, freq = period * 7) # TS object for training data, frequency=number of observations per unit of time.
decomp_ts <- stl(data_ts, s.window = "periodic", robust = TRUE)$time.series 

#Fit Trend and Find Features:
trend_part <- ts(decomp_ts[,2])
trend_fit <- auto.arima(trend_part) # ARIMA
trend_for <- as.vector(forecast(trend_fit, period)$mean) # trend forecast
data_msts <- msts(data_train$value, seasonal.periods = c(period, period*7))#Multi_Seaonal Time Series
# Fourier features to model (Daily and Weekly)
K <- 2
fuur <- fourier(data_msts, K = c(K, K)) 

N <- nrow(data_train)
window <- (N / period) - 1

new_load <- rowSums(decomp_ts[, c(1,3)]) # detrended original time series
lag_seas <- decomp_ts[1:(period*window), 1] # lag feature to model

matrix_train <- data.table(Load = tail(new_load, window*period),
                           fuur[(period + 1):N,],
                           Lag = lag_seas)

# create testing data matrix
test_lag <- decomp_ts[((period*window)+1):N, 1]
fuur_test <- fourier(data_msts, K = c(K, K), h = period)

matrix_test <- data.table(fuur_test,
                          Lag = test_lag)




### Modeling:
N_boot <- 100 # number of bootstraps

pred_mat <- matrix(0, nrow = N_boot, ncol = period)
for(i in 1:N_boot) {
  
  matrixSam <- matrix_train[sample(1:(N-period),
                                   floor((N-period) * sample(seq(0.7, 0.9, by = 0.01), 1)),
                                   replace = TRUE)] # sampling with sampled ratio from 0.7 to 0.9
  tree_bag <- rpart(Load ~ ., data = matrixSam,
                    control = rpart.control(minsplit = sample(2:3, 1),
                                            maxdepth = sample(26:30, 1),
                                            cp = sample(seq(0.0000009, 0.00001, by = 0.0000001), 1)))
  
  # new data and prediction
  pred_mat[i,] <- predict(tree_bag, matrix_test) + mean(trend_for)
}


# Take the median:
pred_melt_rpart <- data.table(melt(pred_mat))
pred_ave_rpart <- pred_melt_rpart[, .(value = median(value)), by = .(Var2)]
pred_ave_rpart[, Var1 := "RPART_Bagg"]



# Plot:
ggplot(pred_melt_rpart, aes(x=Var2, y=value, group = Var1)) +
  geom_line(alpha = 0.75) +
  geom_line(data = pred_ave_rpart, aes(Var2, value), color = "firebrick2", alpha = 0.9, size = 2)



colnames(pred_ave_rpart) = c('x','y','group')

mytest = data_test[,'value']
mytest = as.data.frame(mytest)
mytest['Var'] = seq(1:dim(mytest)[1])
mytest['group'] = 'label'

colnames(mytest) = c('y','x','group')

mytest = mytest[,c('x','y','group')]


final_vis = rbind(pred_ave_rpart,mytest)

ggplot(final_vis,aes(x=x,y=y,group=group,color=group))+geom_line()
