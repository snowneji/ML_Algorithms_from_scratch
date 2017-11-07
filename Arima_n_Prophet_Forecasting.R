require(forecast)
require(ggplot2)
require(reshape2)
require(data.table)
require(prophet)
# Load Data:
data = fread('USD_INR.csv',data.table = F)
data$Date = as.POSIXct(data$Date,format='%B %d, %Y')
data = data[with(data, order(Date)), ]


data = data['Price']
data = ts(data)
#Plot:
plot(data)
#Split:
ntr = as.integer(nrow(data)*0.8)
tr = data[1:ntr]
vl = data[(ntr+1):nrow(data)]
tr = ts(tr)
vl = ts(vl)



#Auto Arima:
model0 = forecast::auto.arima(tr)
pred0 = forecast::forecast(model0,h=length(vl))
pred0 = pred0$mean



#Prophet:
data = fread('USD_INR.csv',data.table = F)
data = data[,c('Price','Date')]
colnames(data) = c('y','ds')
data$ds = as.POSIXct(data$ds,format='%B %d, %Y')
data = data[with(data, order(ds)), ]
data = data[,c('ds','y')]
ntr = as.integer(nrow(data)*0.8)
ptr = data[1:ntr,]
pvl = data[(ntr+1):nrow(data),]
model = prophet(ptr)
pred3 = predict(model,data.frame(ds=pvl$ds))
pred3 = pred3$yhat
pred3 = ts(pred3)
#Plot:
plot(data)
#Split:

#RMSE:
rmse = function(x,y){
  return(sqrt(mean((x-y)**2)))
  
}
print('RMSE:')
print(rmse(as.numeric(vl),as.numeric(pred0)))
print(rmse(as.numeric(vl),as.numeric(pred1)))
print(rmse(as.numeric(vl),as.numeric(pred2)))
print(rmse(as.numeric(vl),as.numeric(pred3)))

#plot:
p_dat1 = data.frame(
label = as.numeric(tr),
arima = as.numeric(tr),
# ets = as.numeric(tr),
# nn = as.numeric(tr),
prophet = as.numeric(tr),
blend = as.numeric(tr) )


p_dat2 = data.frame(
  label=as.numeric(vl),
  arima=as.numeric(pred0),
  # ets=as.numeric(pred1),
  # nn=as.numeric(pred2),
  prophet=as.numeric(pred3),
  blend=(as.numeric(pred0)+as.numeric(pred3))/2
)


p_dat = rbind(p_dat1,p_dat2)
p_dat$id = seq(1:nrow(p_dat))
d = reshape2::melt(p_dat, id="id")





ggplot(data=d,
       aes(x=id, y=value, colour=variable)) +
  geom_line()
