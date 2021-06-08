#Read the data

data = read.csv("USUnemployment.csv", header = T)

head(data)
tail(data)


#Transform the data into a time series
data.ts = ts(c(unname(t(data))[2:13,]), start = c(1948, 1), end = c(2019, 12), frequency = 12 )

plot(data.ts, main = "Monthly Unemployment Rate in USA (1948-2019)", ylab = "Unemployment Rate")

data_2020 = readxl::read_excel('USUnemployment_2020.xlsx')
data.2020.ts = ts(data_2020$`Unemployment Rate`, start = c(2020, 1), frequency = 12)

data.combined = ts(c(data.ts, data.2020.ts), start = start(data.ts), frequency = 12)
plot(data.combined, main = "Monthly Unemployment Rate in USA (1948- Oct 2020)", ylab = "Unemployment Rate")

##############################
#Perform short-term forecasts
##############################

library(forecast)

train = window(data.ts, start = c(2010, 1), end = c(2018, 12))
valid = window(data.ts, start = c(2019, 1))

plot(train)

#ETS model
st.m1 = ets( train, model = 'ZZZ')
st.m1.predict = forecast(st.m1, h = length(valid))

plot(st.m1.predict)
lines(valid)

#ARIMA model

#Remove the trend

train.detrended = diff(train, lag = 1)

plot(train.detrended)

Acf(train.detrended, lag.max = 50)
Pacf(train.detrended, lag.max = 50)

Acf(train.detrended, lag.max = 150)
Pacf(train.detrended, lag.max = 150)

m2 = Arima(train, order = c(3, 1, 1))
m2.forecast = forecast( m2, h = length(valid))

plot(m2.forecast)
lines(valid)

accuracy(st.m1.predict, valid)

accuracy(m2.forecast, valid)

# M2 is more robust. Use M2 to perform short-term forecasts

st.ts = window( data.ts, start = c(2010, 1) )

st.final = Arima( st.ts, order = c(3, 1, 1) )
st.final.forecast = forecast( st.final, h = 12 )
plot(st.final.forecast)

plot(st.final.forecast, ylim = c(2, 16))
lines( data.2020.ts )

##############################
#Perform longterm forecasts
##############################

# Acf(window(data.ts, start = c(1970, 1), end = c(2010, 12)), lag.max = 350)
# Pacf(window(data.ts, start = c(1970, 1), end = c(2010, 12)), lag.max = 250)
# dt = window(data.ts, start = c(1970, 1), end = c(2010, 12))
# m = Arima(dt, order = c(2, 0, 0), seasonal = list(order = c(0, 0, 2), period = 156))
# m.forecast = forecast( m, h = 120)
# plot(m.forecast)

#Aggregate to yearly data
data.agg = aggregate(data.ts, FUN = mean)

train.lt = window(data.agg, start = 1970, end = 2010)
#train.lt = window(data.smooth, end = c(2010, 12))
valid.lt = window(data.agg, start = 2011)

plot(train.lt)

#ETS
lt.m1 = ets( train.lt, model = 'ZZZ')
lt.m1.predict = forecast(lt.m1, h = length(valid.lt))

plot(lt.m1.predict)
lines(valid.lt)

plot(train.lt)


Acf(train.lt, lag.max = 250)
Pacf(train.lt, lag.max = 250)

m3 = Arima(train.lt, order = c(2, 0, 1), seasonal = list(order = c(0,0,2), period = 15) )
m3.forecast = forecast( m3, h = length(valid.lt))

autoplot(m3.forecast)+autolayer(valid.lt)+autolayer(m3$fitted)
accuracy( m3.forecast, valid.lt )

checkresiduals(m3.forecast)

m4 = Arima(window(data.agg, start = 1970), order = c(2, 0, 1), seasonal = list(order = c(0,0,2), period = 15) )
m4.forecast = forecast( m4, h = 10)

plot(m4.forecast)
lines(data.2020.ts)


# Aggregate Quarterly Data

data.quarterly = aggregate(data.ts, nfrequency = 4, FUN = 'mean')

train.lt.q = window(data.quarterly, start = c(1970, 1), end = c(2010, 4) )
valid.lt.q = window(data.quarterly, start = c(2011, 1))

plot(train.lt.q)

Acf(train.lt.q, lag.max = 250)
Pacf(train.lt.q, lag.max = 250)


m5 = Arima( train.lt.q, order = c(2, 0, 0), seasonal = list( order = c(0, 0, 2), period = 52) )
m5.forecast = forecast( m5, h = 40 )

autoplot(m5.forecast)+autolayer(valid.lt.q)+autolayer(m5$fitted)
accuracy(m5.forecast, valid.lt.q)

checkresiduals(m5.forecast)

m6 = Arima(window(data.quarterly, start = c(1970, 1)), order = c(0, 0, 2), seasonal = list(order = c(0,0,2), period = 52) )
m6.forecast = forecast( m6, h = 40)

plot(m6.forecast)
lines(data.2020.ts)



#### Use Neural Networks ####

m7 = nnetar(train.lt.q, repeats = 30, p = 51, P = 2, MaxNWts = 1400)
m7.forecast = forecast(m7, h = 40)
autoplot(m7.forecast)+autolayer(valid.lt.q)+autolayer(m7$fitted)

m8 = nnetar(train.lt.q)
m8.forecast = forecast(m8, h = 40)
autoplot(m8.forecast)+autolayer(valid.lt.q)+autolayer(m8$fitted)

checkresiduals(m8.forecast)


########################
#External Factors
########################

####PCE####

library(astsa)
dq = aggregate(data.combined, nfrequency = 4, FUN = 'mean')

pce = readxl::read_excel('PCECC96.xls')

pce.ts = ts(pce$PCECC96, start = c(1947, 1), frequency = 4 )
pce.dt = window(pce.ts, start = c(1948, 1))
plot(pce.dt)

lag2.plot(pce.ts, dq, 24)
Ccf(pce.dt, dq, 26)

newdata = ts.intersect( UR = dq, Pce = pce.dt, lagPce10 = lag(pce.dt, -10), lagUR52 = lag(dq, -54) )
head(newdata)

m9 = tslm(UR ~ Pce + lagPce10 + lagUR52, data = window(newdata, start = c(1970,1))) 
summary(m9)

plot(m9$fitted.values, ylim = c(3, 12), col = 'red')+lines(window(dq), start = c(1970,1))


####Durable Goods Orders####

dg = readxl::read_excel('DGORDER.xls')

dg.ts = ts(dg$DGORDER, start = c(1992, 2), frequency = 12)
data.dg = window(data.combined, start = c(1992, 2))
plot(log(data.dg))
lag2.plot(dg.ts, log(data.dg), 10)
Ccf(dg.ts, data.dg, 26)

newdata_2 = ts.intersect( UR = log(data.dg), DG = dg.ts, lagDG1 = lag(dg.ts, -1) )
head(newdata_2)
m10 = tslm(UR ~ DG + lagDG1 , data = newdata_2 )
summary(m10)
plot(m10$fitted.values, col = 'red', ylim = c(0,3))+lines(window(log(data.dg)), start = c(1992,1))


####Dow-Jones Industrial Average####

dju = read.csv('^DJU.csv')
dju.ts = ts(dju$Adj.Close, start = c(1992, 1), end = c(2020, 10), frequency = 12)
plot(dju.ts)

#De-trend DJU
dju.ts.diff = diff(dju.ts, lag = 1)
plot(dju.ts.diff)

data.dju = window(data.combined, start = c(1992, 1), end = c(2020, 10))

lag2.plot(dju.ts, data.dju, 25)
Ccf(dju.ts, data.dju, 60)

lag2.plot(dju.ts.diff, data.dju, 25)
Ccf(dju.ts.diff, data.dju, 60)

newdata_3 = ts.intersect( UR = data.dju, DJU = dju.ts, lagDJU48 = lag(dju.ts, -48), lagUR216 = lag(data.combined, -216)  )  #, DJUDiff = dju.ts.diff, lagDJUDiff1 = lag(dju.ts.diff, -1), lagDJUDiff29 = lag(dju.ts.diff, -29) )
head(newdata_3)

m11 = tslm(UR ~ DJU + lagDJU48 + lagUR216, data = newdata_3 ) # + DJUDiff + lagDJUDiff1 + lagDJUDiff29, data = newdata_3 )
summary(m11)
plot(m11$fitted.values, col = 'red', ylim = c(3,15))+lines(window(data.dju), start = c(1992,1))
