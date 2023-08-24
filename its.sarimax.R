# load the packages
library(astsa)
library(forecast)
library(dplyr)
library(zoo)
library(lubridate)

# read data from csv file
setwd("D:/RA/Time series/apple shipment price")
data<-read.csv('data-interppolated.csv', fileEncoding = 'UTF-8-BOM')

# Convert the "date" column to a date format
data$date <- as.Date(data$date)
# Create new columns for year and month
data$year <- year(data$date)
data$month <- month(data$date)
data$day <- day(data$date)
head(data)


Gala=na.omit(data[c('date', 'Gala','year','month','day')])
head(Gala)
HoneyCrisp=na.omit(data[c('date', 'HoneyCrisp','year','month','day')])
CrippsPink=na.omit(data[c('date', 'CrippsPink','year','month','day')])
RedDelicious=na.omit(data[c('date', 'RedDelicious','year','month','day')])
Fuji=na.omit(data[c('date', 'Fuji','year','month','day')])
GoldenDelicious=na.omit(data[c('date', 'GoldenDelicious','year','month','day')])
GrannySmith=na.omit(data[c('date', 'GrannySmith','year','month','day')])
Braeburn=na.omit(data[c('date', 'Braeburn','year','month','day')])
CosmicCrisp=na.omit(data[c('date', 'CosmicCrisp','year','month','day')])

## event start date
# Gala	2017-08-25
# HoneyCrisp		2017-08-21
# CrippsPink	 	2017-10-23
# RedDelicious	2017-10-02
# Fuji	2017-09-25
# GoldenDelicious	2017-08-21
# GrannySmith	2017-09-18
# Braeburn  2017-10-09

### ordinary least squares (linear) regression for continuous outcomes

## variables
##generate week to caputure the seasonality
## generate a dummy variable (X) indicating the pre and post period
## pre-intervention period (coded 0) or the post-intervention period (coded 1);
## generate time elapse (T)


## ---------------------------------------------------------------------------##
## -------------------------------- Gala -------------------------------------##
## ---------------------------------------------------------------------------##
# Convert data to time series object
Gala.ts <-ts(Gala[,2], freq=365.25/7, start=decimal_date(ymd("2008-08-25")))
Gala.ts

# Plot data to visualize time series
options(scipen=5)
plot(Gala.ts, ylim=c(0,1000), type='l', col="blue", xlab="Month", ylab="Dispensings")
## Convert the date "2017-08-25" to the corresponding week number
intervention_date_Gala <- as.Date("2017-08-25")
weeks_from_start_Gala <- (decimal_date(intervention_date)) 
abline(v=weeks_from_start_Gala, col="gray", lty="dashed", lwd=2)

# View ACF/PACF plots of undifferenced data
acf(Gala.ts, max.lag=24)

# View ACF/PACF plots of differenced/seasonally differenced data
acf(diff(diff(Gala.ts,12)), max.lag=24)

## generate a dummy variable (X) indicating the pre and post period
## pre-intervention period (coded 0) or the post-intervention period (coded 1)
# Create variable representing step change and view
step_Gala <- ifelse(Gala$date >= as.Date('2017-08-21'), 1, 0)
step_Gala

### Post.intervention.time = A number, the time elapsed since the Intervention
# Create variable representing ramp (change in slope) and view
ramp_Gala = c(rep(0,469),1:(758-469))
ramp_Gala

# Use automated algorithm to identify p/q parameters
# Specify first difference = 1 and seasonal difference = 1
model1 <- auto.arima(Gala.ts, seasonal=TRUE, xreg=cbind(step_Gala,ramp_Gala), max.d=1, max.D=1, stepwise=FALSE, trace=TRUE)
model1_Gala <-model1
## Best model: Regression with ARIMA(1,0,2)(2,1,0)[52] errors 

## significance level
options(scipen = 999)
(1-pnorm(abs(model1_Gala$coef)/sqrt(diag(model1_Gala$var.coef))))*2

# Check residuals
checkresiduals(model1_Gala)
Box.test(model1_Gala$residuals, lag = 24, type = "Ljung-Box")

# Estimate parameters and confidence intervals
summary(model1_Gala)
confint(model1_Gala)
## autocorrelated residuals

# To forecast the counterfactual, model data excluding post-intervention time period
model2_Gala <- Arima(window(Gala.ts, end=decimal_date(as.Date('2017-08-18'))), order=c(1,0,2), seasonal=list(order=c(2,1,0), period=365.25/7))

# Forecast 12 months post-intervention and convert to time series object
fc_Gala <- forecast(model2_Gala, h=288)
fc_Gala.values=as.numeric(fc_Gala$mean)
fc_Gala.values = ifelse(fc_Gala.values>0,fc_Gala.values,0)
fc.ts_Gala <- ts(fc_Gala.values, start=time(Gala.ts)[length(Gala.ts)-288], frequency=365.25/7)

fc.ts_Gala
Gala.ts
# Combine with observed data
Gala.ts.2 <- ts.union(Gala.ts, fc.ts_Gala)
Gala.ts.2

# Plot
plot(Gala.ts.2, type="l", plot.type="s", col=c('blue','red'), xlab="Month", ylab="Dispensings", linetype=c("solid","dashed"), ylim=c(0,1000))
abline(v=time(Gala.ts)[length(Gala.ts)-288], lty="dashed", col="gray")

Gala.ts_values = as.numeric(Gala.ts)
fc.ts_Gala_values = as.numeric(fc.ts_Gala)

# Define the frequency (52 for weekly data) and the start date
frequency <- 52
start_date <- as.Date("2008-08-25")
start_date2 <- as.Date("2017-08-25")

# Convert the numeric array to a weekly time series object
my_ts1 <- ts(Gala.ts_values, frequency = frequency, start = c(year(start_date), week(start_date)))
my_ts2 <- ts(fc.ts_Gala_values, frequency = frequency, start = c(year(start_date2), week(start_date)))
my_ts.2 <- ts.union(my_ts1, my_ts2)
plot(my_ts.2, type="l", plot.type="s", col=c('blue','red'), xlab="Month", ylab="Dispensings", linetype=c("solid","dashed"), ylim=c(0,1000))
abline(v=weeks_from_start_Gala, lty="dashed", col="gray")
title("Gala shipments, 2008-2023")


## ---------------------------------------------------------------------------##
## -------------------------------- HoneyCrisp -------------------------------##
## ---------------------------------------------------------------------------##
# Convert data to time series object
HoneyCrisp.ts <-ts(HoneyCrisp[,2], freq=365.25/7, start=decimal_date(ymd("2008-09-01")))
HoneyCrisp.ts
summary(HoneyCrisp.ts)

# Plot data to visualize time series
options(scipen=5)
plot(HoneyCrisp.ts, ylim=c(0,1000), type='l', col="blue", xlab="Month", ylab="Dispensings")
intervention_date_HoneyCrisp <- as.Date("2017-08-21")
weeks_from_start_HoneyCrisp <- (decimal_date(intervention_date)) 
abline(v=weeks_from_start_HoneyCrisp, col="gray", lty="dashed", lwd=2)

# View ACF/PACF plots of undifferenced data
acf(HoneyCrisp.ts, max.lag=24)

# View ACF/PACF plots of differenced/seasonally differenced data
acf(diff(diff(HoneyCrisp.ts,12)), max.lag=24)

## generate a dummy variable (X) indicating the pre and post period
## pre-intervention period (coded 0) or the post-intervention period (coded 1)
# Create variable representing step change and view
step_HoneyCrisp <- ifelse(HoneyCrisp$date >= as.Date('2017-08-21'), 1, 0)
step_HoneyCrisp

### Post.intervention.time = A number, the time elapsed since the Intervention
# Create variable representing ramp (change in slope) and view
ramp_HoneyCrisp = c(rep(0,469),1:(757-469))
ramp_HoneyCrisp

# Use automated algorithm to identify p/q parameters
# Specify first difference = 1 and seasonal difference = 1
model1 <- auto.arima(HoneyCrisp.ts, seasonal=TRUE, xreg=cbind(step_HoneyCrisp,ramp_HoneyCrisp), max.d=1, max.D=1, stepwise=FALSE, trace=TRUE)
model1_HoneyCrisp <-model1
## Best model: Regression with ARIMA(3,1,1)(0,1,1)[52] errors 

# Check residuals
checkresiduals(model1_HoneyCrisp)
Box.test(model1_HoneyCrisp$residuals, lag = 24, type = "Ljung-Box")

# Estimate parameters and confidence intervals
summary(model1_HoneyCrisp)
confint(model1_HoneyCrisp)
## significance level
options(scipen = 999)
(1-pnorm(abs(model1_HoneyCrisp$coef)/sqrt(diag(model1_HoneyCrisp$var.coef))))*2
## autocorrelated residuals

# To forecast the counterfactual, model data excluding post-intervention time period
model2_HoneyCrisp <- Arima(window(HoneyCrisp.ts, end=decimal_date(as.Date('2017-08-18'))), order=c(3,1,1), seasonal=list(order=c(0,1,1), period=365.25/7))

# Forecast 12 months post-intervention and convert to time series object
fc_HoneyCrisp <- forecast(model2_HoneyCrisp, h=287)
fc_HoneyCrisp.values=as.numeric(fc_HoneyCrisp$mean)
fc_HoneyCrisp.values = ifelse(fc_HoneyCrisp.values>0,fc_HoneyCrisp.values,0)
fc.ts_HoneyCrisp <- ts(fc_HoneyCrisp.values, start=time(HoneyCrisp.ts)[length(HoneyCrisp.ts)-287], frequency=365.25/7)

# Combine with observed data
HoneyCrisp.ts.2 <- ts.union(HoneyCrisp.ts, fc.ts_HoneyCrisp)

# Plot
plot(HoneyCrisp.ts.2, type="l", plot.type="s", col=c('blue','red'), xlab="Month", ylab="Dispensings", linetype=c("solid","dashed"), ylim=c(0,1000))
abline(v=time(HoneyCrisp.ts)[length(HoneyCrisp.ts)-288], lty="dashed", col="gray")

HoneyCrisp.ts_values = as.numeric(HoneyCrisp.ts)
fc.ts_HoneyCrisp_values = as.numeric(fc.ts_HoneyCrisp)

# Define the frequency (52 for weekly data) and the start date
frequency <- 52
start_date <- as.Date("2008-09-01")
start_date2 <- as.Date("2017-08-25")

# Convert the numeric array to a weekly time series object
my_ts1 <- ts(HoneyCrisp.ts_values, frequency = frequency, start = c(year(start_date), week(start_date)))
my_ts2 <- ts(fc.ts_HoneyCrisp_values, frequency = frequency, start = c(year(start_date2), week(start_date)))
my_ts.2 <- ts.union(my_ts1, my_ts2)
plot(my_ts.2, type="l", plot.type="s", col=c('blue','red'), xlab="Month", ylab="Dispensings", linetype=c("solid","dashed"), ylim=c(0,600))
abline(v=weeks_from_start_HoneyCrisp, lty="dashed", col="gray")
title("HoneyCrisp shipments, 2008-2023")



## ---------------------------------------------------------------------------##
## -------------------------------- CrippsPink -------------------------------##
## ---------------------------------------------------------------------------##
# Convert data to time series object
CrippsPink.ts <-ts(CrippsPink[,2], freq=365.25/7, start=decimal_date(ymd("2008-10-27")))
CrippsPink.ts

# Plot data to visualize time series
options(scipen=5)
plot(CrippsPink.ts, ylim=c(0,1000), type='l', col="blue", xlab="Month", ylab="Dispensings")
intervention_date_CrippsPink <- as.Date("2017-08-21")
weeks_from_start_CrippsPink <- (decimal_date(intervention_date)) 
abline(v=weeks_from_start_CrippsPink, col="gray", lty="dashed", lwd=2)

# View ACF/PACF plots of undifferenced data
acf(CrippsPink.ts, max.lag=24)

# View ACF/PACF plots of differenced/seasonally differenced data
acf(diff(diff(CrippsPink.ts,12)), max.lag=24)

## generate a dummy variable (X) indicating the pre and post period
## pre-intervention period (coded 0) or the post-intervention period (coded 1)
# Create variable representing step change and view
step_CrippsPink <- ifelse(CrippsPink$date >= as.Date('2017-10-23'), 1, 0)
step_CrippsPink

### Post.intervention.time = A number, the time elapsed since the Intervention
# Create variable representing ramp (change in slope) and view
ramp_CrippsPink = c(rep(0,479),1:(749-479))
ramp_CrippsPink

# Use automated algorithm to identify p/q parameters
# Specify first difference = 1 and seasonal difference = 1
model1_CrippsPink <- auto.arima(CrippsPink.ts, seasonal=TRUE, xreg=cbind(step_CrippsPink,ramp_CrippsPink), max.d=1, max.D=1, stepwise=FALSE, trace=TRUE)
## Best model: Regression with ARIMA(2,0,1)(1,1,1)[52] errors 

# Check residuals
checkresiduals(model1_CrippsPink)
Box.test(model1_CrippsPink$residuals, lag = 24, type = "Ljung-Box")

# Estimate parameters and confidence intervals
summary(model1_CrippsPink)
confint(model1_CrippsPink)
## significance level
options(scipen = 999)
(1-pnorm(abs(model1_CrippsPink$coef)/sqrt(diag(model1_CrippsPink$var.coef))))*2
## autocorrelated residuals

# To forecast the counterfactual, model data excluding post-intervention time period
model2_CrippsPink <- Arima(window(CrippsPink.ts, end=decimal_date(as.Date('2017-10-23'))), order=c(2,0,1), seasonal=list(order=c(1,1,1), period=365.25/7))

# Forecast 12 months post-intervention and convert to time series object
fc_CrippsPink <- forecast(model2_CrippsPink, h=269)
fc_CrippsPink.values=as.numeric(fc_CrippsPink$mean)
fc_CrippsPink.values = ifelse(fc_CrippsPink.values>0,fc_CrippsPink.values,0)
fc.ts_CrippsPink <- ts(fc_CrippsPink.values, start=time(CrippsPink.ts)[length(CrippsPink.ts)-269], frequency=365.25/7)

# Combine with observed data
CrippsPink.ts.2 <- ts.union(CrippsPink.ts, fc.ts_CrippsPink)

# Plot
plot(CrippsPink.ts.2, type="l", plot.type="s", col=c('blue','red'), xlab="Month", ylab="Dispensings", linetype=c("solid","dashed"), ylim=c(0,1000))
abline(v=time(CrippsPink.ts)[length(CrippsPink.ts)-269], lty="dashed", col="gray")

CrippsPink.ts_values = as.numeric(CrippsPink.ts)
fc.ts_CrippsPink_values = as.numeric(fc.ts_CrippsPink)

# Define the frequency (52 for weekly data) and the start date
frequency <- 52
start_date <- as.Date("2008-09-01")
start_date2 <- as.Date("2017-08-25")

# Convert the numeric array to a weekly time series object
my_ts1 <- ts(CrippsPink.ts_values, frequency = frequency, start = c(year(start_date), week(start_date)))
my_ts2 <- ts(fc.ts_CrippsPink_values, frequency = frequency, start = c(year(start_date2), week(start_date)))
my_ts.2 <- ts.union(my_ts1, my_ts2)
plot(my_ts.2, type="l", plot.type="s", col=c('blue','red'), xlab="Month", ylab="Dispensings", linetype=c("solid","dashed"), ylim=c(0,250))
abline(v=weeks_from_start_CrippsPink, lty="dashed", col="gray")
title("CrippsPink shipments, 2008-2023")


## ---------------------------------------------------------------------------##
## -------------------------------- RedDelicious -------------------------------##
## ---------------------------------------------------------------------------##
# Convert data to time series object
RedDelicious.ts <-ts(RedDelicious[,2], freq=365.25/7, start=decimal_date(ymd("2008-09-15")))
RedDelicious.ts

# Plot data to visualize time series
options(scipen=5)
plot(RedDelicious.ts, ylim=c(0,1200), type='l', col="blue", xlab="Month", ylab="Dispensings")
intervention_date_RedDelicious <- as.Date("2017-10-02")
weeks_from_start_RedDelicious <- (decimal_date(intervention_date)) 
abline(v=weeks_from_start_RedDelicious, col="gray", lty="dashed", lwd=2)

# View ACF/PACF plots of undifferenced data
acf(RedDelicious.ts, max.lag=24)

# View ACF/PACF plots of differenced/seasonally differenced data
acf(diff(diff(RedDelicious.ts,12)), max.lag=24)

## generate a dummy variable (X) indicating the pre and post period
## pre-intervention period (coded 0) or the post-intervention period (coded 1)
# Create variable representing step change and view
step_RedDelicious <- ifelse(RedDelicious$date >= as.Date('2017-10-02'), 1, 0)
step_RedDelicious

### Post.intervention.time = A number, the time elapsed since the Intervention
# Create variable representing ramp (change in slope) and view
ramp_RedDelicious = c(rep(0,472),1:(755-472))
ramp_RedDelicious

# Use automated algorithm to identify p/q parameters
# Specify first difference = 1 and seasonal difference = 1
model1_RedDelicious <- auto.arima(RedDelicious.ts, seasonal=TRUE, xreg=cbind(step_RedDelicious,ramp_RedDelicious), max.d=1, max.D=1, stepwise=FALSE, trace=TRUE)
## Best model: Regression with ARIMA(1,0,2)(2,1,0)[52] errors 

# Check residuals
checkresiduals(model1_RedDelicious)
Box.test(model1_RedDelicious$residuals, lag = 24, type = "Ljung-Box")

# Estimate parameters and confidence intervals
summary(model1_RedDelicious)
confint(model1_RedDelicious)
## significance level
options(scipen = 999)
(1-pnorm(abs(model1_RedDelicious$coef)/sqrt(diag(model1_RedDelicious$var.coef))))*2
## autocorrelated residuals

# To forecast the counterfactual, model data excluding post-intervention time period
model2_RedDelicious <- Arima(window(RedDelicious.ts, end=decimal_date(as.Date('2017-10-02'))), order=c(1,0,2), seasonal=list(order=c(2,1,0), period=365.25/7))

# Forecast 12 months post-intervention and convert to time series object
fc_RedDelicious <- forecast(model2_RedDelicious, h=282)
fc_RedDelicious.values=as.numeric(fc_RedDelicious$mean)
fc_RedDelicious.values = ifelse(fc_RedDelicious.values>0,fc_RedDelicious.values,0)
fc.ts_RedDelicious <- ts(fc_RedDelicious.values, start=time(RedDelicious.ts)[length(RedDelicious.ts)-282], frequency=365.25/7)

# Combine with observed data
RedDelicious.ts.2 <- ts.union(RedDelicious.ts, fc.ts_RedDelicious)

# Plot
plot(RedDelicious.ts.2, type="l", plot.type="s", col=c('blue','red'), xlab="Month", ylab="Dispensings", linetype=c("solid","dashed"), ylim=c(0,1200))
abline(v=time(RedDelicious.ts)[length(RedDelicious.ts)-282], lty="dashed", col="gray")

RedDelicious.ts_values = as.numeric(RedDelicious.ts)
fc.ts_RedDelicious_values = as.numeric(fc.ts_RedDelicious)

# Define the frequency (52 for weekly data) and the start date
frequency <- 52
start_date <- as.Date("2008-09-01")
start_date2 <- as.Date("2017-08-25")

# Convert the numeric array to a weekly time series object
my_ts1 <- ts(RedDelicious.ts_values, frequency = frequency, start = c(year(start_date), week(start_date)))
my_ts2 <- ts(fc.ts_RedDelicious_values, frequency = frequency, start = c(year(start_date2), week(start_date)))
my_ts.2 <- ts.union(my_ts1, my_ts2)
plot(my_ts.2, type="l", plot.type="s", col=c('blue','red'), xlab="Month", ylab="Dispensings", linetype=c("solid","dashed"), ylim=c(0,1200))
abline(v=weeks_from_start_RedDelicious, lty="dashed", col="gray")
title("RedDelicious shipments, 2008-2023")



## ---------------------------------------------------------------------------##
## -------------------------------- Fuji -------------------------------##
## ---------------------------------------------------------------------------##
# Convert data to time series object
Fuji.ts <-ts(Fuji[,2], freq=365.25/7, start=decimal_date(ymd("2008-09-08")))
Fuji.ts

# Plot data to visualize time series
options(scipen=5)
plot(Fuji.ts, ylim=c(0,800), type='l', col="blue", xlab="Month", ylab="Dispensings")
intervention_date_Fuji <- as.Date("2017-09-25")
weeks_from_start_Fuji <- (decimal_date(intervention_date)) 
abline(v=weeks_from_start_Fuji, col="gray", lty="dashed", lwd=2)

# View ACF/PACF plots of undifferenced data
acf(Fuji.ts, max.lag=24)

# View ACF/PACF plots of differenced/seasonally differenced data
acf(diff(diff(Fuji.ts,12)), max.lag=24)

## generate a dummy variable (X) indicating the pre and post period
## pre-intervention period (coded 0) or the post-intervention period (coded 1)
# Create variable representing step change and view
step_Fuji <- ifelse(Fuji$date >= as.Date('2017-09-25'), 1, 0)
step_Fuji

### Post.intervention.time = A number, the time elapsed since the Intervention
# Create variable representing ramp (change in slope) and view
ramp_Fuji = c(rep(0,473),1:(756-473))
ramp_Fuji

# Use automated algorithm to identify p/q parameters
# Specify first difference = 1 and seasonal difference = 1
model1_Fuji <- auto.arima(Fuji.ts, seasonal=TRUE, xreg=cbind(step_Fuji,ramp_Fuji), max.d=1, max.D=1, stepwise=FALSE, trace=TRUE)
## Best model: Regression with ARIMA(1,0,2)(1,1,1)[52] errors 

# Check residuals
checkresiduals(model1_Fuji)
Box.test(model1_Fuji$residuals, lag = 24, type = "Ljung-Box")

# Estimate parameters and confidence intervals
summary(model1_Fuji)
confint(model1_Fuji)
## significance level
options(scipen = 999)
(1-pnorm(abs(model1_Fuji$coef)/sqrt(diag(model1_Fuji$var.coef))))*2
## autocorrelated residuals

# To forecast the counterfactual, model data excluding post-intervention time period
model2_Fuji <- Arima(window(Fuji.ts, end=decimal_date(as.Date('2017-09-25'))), order=c(1,0,2), seasonal=list(order=c(1,1,1), period=365.25/7))

# Forecast 12 months post-intervention and convert to time series object
fc_Fuji <- forecast(model2_Fuji, h=282)
fc_Fuji.values=as.numeric(fc_Fuji$mean)
fc_Fuji.values = ifelse(fc_Fuji.values>0,fc_Fuji.values,0)
fc.ts_Fuji <- ts(fc_Fuji.values, start=time(Fuji.ts)[length(Fuji.ts)-282], frequency=365.25/7)

# Combine with observed data
Fuji.ts.2 <- ts.union(Fuji.ts, fc.ts_Fuji)

# Plot
plot(Fuji.ts.2, type="l", plot.type="s", col=c('blue','red'), xlab="Month", ylab="Dispensings", linetype=c("solid","dashed"), ylim=c(0,800))
abline(v=time(Fuji.ts)[length(Fuji.ts)-282], lty="dashed", col="gray")

Fuji.ts_values = as.numeric(Fuji.ts)
fc.ts_Fuji_values = as.numeric(fc.ts_Fuji)

# Define the frequency (52 for weekly data) and the start date
frequency <- 52
start_date <- as.Date("2008-09-01")
start_date2 <- as.Date("2017-08-25")

# Convert the numeric array to a weekly time series object
my_ts1 <- ts(Fuji.ts_values, frequency = frequency, start = c(year(start_date), week(start_date)))
my_ts2 <- ts(fc.ts_Fuji_values, frequency = frequency, start = c(year(start_date2), week(start_date)))
my_ts.2 <- ts.union(my_ts1, my_ts2)
plot(my_ts.2, type="l", plot.type="s", col=c('blue','red'), xlab="Month", ylab="Dispensings", linetype=c("solid","dashed"), ylim=c(0,800))
abline(v=weeks_from_start_Fuji, lty="dashed", col="gray")
title("Fuji shipments, 2008-2023")


## ---------------------------------------------------------------------------##
## -------------------------------- GoldenDelicious -------------------------------##
## ---------------------------------------------------------------------------##
# Convert data to time series object
GoldenDelicious.ts <-ts(GoldenDelicious[,2], freq=365.25/7, start=decimal_date(ymd("2008-09-08")))
GoldenDelicious.ts

# Plot data to visualize time series
options(scipen=5)
plot(GoldenDelicious.ts, ylim=c(0,500), type='l', col="blue", xlab="Month", ylab="Dispensings")
intervention_date_GoldenDelicious <- as.Date("2017-09-25")
weeks_from_start_GoldenDelicious <- (decimal_date(intervention_date)) 
abline(v=weeks_from_start_GoldenDelicious, col="gray", lty="dashed", lwd=2)

# View ACF/PACF plots of undifferenced data
acf(GoldenDelicious.ts, max.lag=24)

# View ACF/PACF plots of differenced/seasonally differenced data
acf(diff(diff(GoldenDelicious.ts,12)), max.lag=24)

## generate a dummy variable (X) indicating the pre and post period
## pre-intervention period (coded 0) or the post-intervention period (coded 1)
# Create variable representing step change and view
step_GoldenDelicious <- ifelse(GoldenDelicious$date >= as.Date('2017-08-21'), 1, 0)
step_GoldenDelicious

### Post.intervention.time = A number, the time elapsed since the Intervention
# Create variable representing ramp (change in slope) and view
ramp_GoldenDelicious = c(rep(0,468),1:(756-468))
ramp_GoldenDelicious

# Use automated algorithm to identify p/q parameters
# Specify first difference = 1 and seasonal difference = 1
model1_GoldenDelicious <- auto.arima(GoldenDelicious.ts, seasonal=TRUE, xreg=cbind(step_GoldenDelicious,ramp_GoldenDelicious), max.d=1, max.D=1, stepwise=FALSE, trace=TRUE)
## Best model: Regression with ARIMA(1,0,2)(2,1,0)[52] errors 

# Check residuals
checkresiduals(model1_GoldenDelicious)
Box.test(model1_GoldenDelicious$residuals, lag = 24, type = "Ljung-Box")

# Estimate parameters and confidence intervals
summary(model1_GoldenDelicious)
confint(model1_GoldenDelicious)
## significance level
options(scipen = 999)
(1-pnorm(abs(model1_GoldenDelicious$coef)/sqrt(diag(model1_GoldenDelicious$var.coef))))*2
## autocorrelated residuals

# To forecast the counterfactual, model data excluding post-intervention time period
model2_GoldenDelicious <- Arima(window(GoldenDelicious.ts, end=decimal_date(as.Date('2017-08-21'))), order=c(1,0,2), seasonal=list(order=c(2,1,0), period=365.25/7))

# Forecast 12 months post-intervention and convert to time series object
fc_GoldenDelicious <- forecast(model2_GoldenDelicious, h=287)
fc_GoldenDelicious.values=as.numeric(fc_GoldenDelicious$mean)
fc_GoldenDelicious.values = ifelse(fc_GoldenDelicious.values>0,fc_GoldenDelicious.values,0)
fc.ts_GoldenDelicious <- ts(fc_GoldenDelicious.values, start=time(GoldenDelicious.ts)[length(GoldenDelicious.ts)-287], frequency=365.25/7)

# Combine with observed data
GoldenDelicious.ts.2 <- ts.union(GoldenDelicious.ts, fc.ts_GoldenDelicious)

# Plot
plot(GoldenDelicious.ts.2, type="l", plot.type="s", col=c('blue','red'), xlab="Month", ylab="Dispensings", linetype=c("solid","dashed"), ylim=c(0,500))
abline(v=time(GoldenDelicious.ts)[length(GoldenDelicious.ts)-287], lty="dashed", col="gray")

GoldenDelicious.ts_values = as.numeric(GoldenDelicious.ts)
fc.ts_GoldenDelicious_values = as.numeric(fc.ts_GoldenDelicious)

# Define the frequency (52 for weekly data) and the start date
frequency <- 52
start_date <- as.Date("2008-09-08")
start_date2 <- as.Date("2017-08-21")

# Convert the numeric array to a weekly time series object
my_ts1 <- ts(GoldenDelicious.ts_values, frequency = frequency, start = c(year(start_date), week(start_date)))
my_ts2 <- ts(fc.ts_GoldenDelicious_values, frequency = frequency, start = c(year(start_date2), week(start_date)))
my_ts.2 <- ts.union(my_ts1, my_ts2)
plot(my_ts.2, type="l", plot.type="s", col=c('blue','red'), xlab="Month", ylab="Dispensings", linetype=c("solid","dashed"), ylim=c(0,500))
abline(v=weeks_from_start_GoldenDelicious, lty="dashed", col="gray")
title("GoldenDelicious shipments, 2008-2023")


## ---------------------------------------------------------------------------##
## -------------------------------- GrannySmith -------------------------------##
## ---------------------------------------------------------------------------##
# Convert data to time series object
GrannySmith.ts <-ts(GrannySmith[,2], freq=365.25/7, start=decimal_date(ymd("2008-09-08")))
GrannySmith.ts

# Plot data to visualize time series
options(scipen=5)
plot(GrannySmith.ts, ylim=c(0,200), type='l', col="blue", xlab="Month", ylab="Dispensings")
intervention_date_GrannySmith <- as.Date("2017-09-18")
weeks_from_start_GrannySmith <- (decimal_date(intervention_date_GrannySmith)) 
abline(v=weeks_from_start_GrannySmith, col="gray", lty="dashed", lwd=2)

# View ACF/PACF plots of undifferenced data
acf(GrannySmith.ts, max.lag=24)

# View ACF/PACF plots of differenced/seasonally differenced data
acf(diff(diff(GrannySmith.ts,12)), max.lag=24)

## generate a dummy variable (X) indicating the pre and post period
## pre-intervention period (coded 0) or the post-intervention period (coded 1)
# Create variable representing step change and view
step_GrannySmith <- ifelse(GrannySmith$date >= as.Date('2017-09-18'), 1, 0)
step_GrannySmith

### Post.intervention.time = A number, the time elapsed since the Intervention
# Create variable representing ramp (change in slope) and view
ramp_GrannySmith = c(rep(0,472),1:(756-472))
ramp_GrannySmith

# Use automated algorithm to identify p/q parameters
# Specify first difference = 1 and seasonal difference = 1
model1_GrannySmith <- auto.arima(GrannySmith.ts, seasonal=TRUE, xreg=cbind(step_GrannySmith,ramp_GrannySmith), max.d=1, max.D=1, stepwise=FALSE, trace=TRUE)
## Best model: Regression with ARIMA(4,0,0)(0,1,1)[52] errors 

# Check residuals
checkresiduals(model1_GrannySmith)
Box.test(model1_GrannySmith$residuals, lag = 24, type = "Ljung-Box")

# Estimate parameters and confidence intervals
summary(model1_GrannySmith)
confint(model1_GrannySmith)
## significance level
options(scipen = 999)
(1-pnorm(abs(model1_GrannySmith$coef)/sqrt(diag(model1_GrannySmith$var.coef))))*2
## autocorrelated residuals

# To forecast the counterfactual, model data excluding post-intervention time period
model2_GrannySmith <- Arima(window(GrannySmith.ts, end=decimal_date(as.Date('2017-09-18'))), order=c(4,0,0), seasonal=list(order=c(0,1,1), period=365.25/7))

# Forecast 12 months post-intervention and convert to time series object
fc_GrannySmith <- forecast(model2_GrannySmith, h=284)
fc_GrannySmith.values=as.numeric(fc_GrannySmith$mean)
fc_GrannySmith.values = ifelse(fc_GrannySmith.values>0,fc_GrannySmith.values,0)
fc.ts_GrannySmith <- ts(fc_GrannySmith.values, start=time(GrannySmith.ts)[length(GrannySmith.ts)-284], frequency=365.25/7)

# Combine with observed data
GrannySmith.ts.2 <- ts.union(GrannySmith.ts, fc.ts_GrannySmith)

# Plot
plot(GrannySmith.ts.2, type="l", plot.type="s", col=c('blue','red'), xlab="Month", ylab="Dispensings", linetype=c("solid","dashed"), ylim=c(0,500))
abline(v=time(GrannySmith.ts)[length(GrannySmith.ts)-284], lty="dashed", col="gray")

GrannySmith.ts_values = as.numeric(GrannySmith.ts)
fc.ts_GrannySmith_values = as.numeric(fc.ts_GrannySmith)

# Define the frequency (52 for weekly data) and the start date
frequency <- 52
start_date <- as.Date("2008-09-08")
start_date2 <- as.Date("2017-09-18")

# Convert the numeric array to a weekly time series object
my_ts1 <- ts(GrannySmith.ts_values, frequency = frequency, start = c(year(start_date), week(start_date)))
my_ts2 <- ts(fc.ts_GrannySmith_values, frequency = frequency, start = c(year(start_date2), week(start_date)))
my_ts.2 <- ts.union(my_ts1, my_ts2)
plot(my_ts.2, type="l", plot.type="s", col=c('blue','red'), xlab="Month", ylab="Dispensings", linetype=c("solid","dashed"), ylim=c(0,200))
abline(v=weeks_from_start_GrannySmith, lty="dashed", col="gray")
title("GrannySmith shipments, 2008-2023")




## ---------------------------------------------------------------------------##
## -------------------------------- Braeburn -------------------------------##
## ---------------------------------------------------------------------------##
# Convert data to time series object
Braeburn.ts <-ts(Braeburn[,2], freq=365.25/7, start=decimal_date(ymd("2008-09-22")))
Braeburn.ts

# Plot data to visualize time series
options(scipen=5)
plot(Braeburn.ts, ylim=c(0,200), type='l', col="blue", xlab="Month", ylab="Dispensings")
intervention_date_Braeburn <- as.Date("2017-10-09")
weeks_from_start_Braeburn <- (decimal_date(intervention_date)) 
abline(v=weeks_from_start_Braeburn, col="gray", lty="dashed", lwd=2)

# View ACF/PACF plots of undifferenced data
acf(Braeburn.ts, max.lag=24)

# View ACF/PACF plots of differenced/seasonally differenced data
acf(diff(diff(Braeburn.ts,12)), max.lag=24)

## generate a dummy variable (X) indicating the pre and post period
## pre-intervention period (coded 0) or the post-intervention period (coded 1)
# Create variable representing step change and view
step_Braeburn <- ifelse(Braeburn$date >= as.Date('2017-10-09'), 1, 0)
step_Braeburn

### Post.intervention.time = A number, the time elapsed since the Intervention
# Create variable representing ramp (change in slope) and view
ramp_Braeburn = c(rep(0,473),1:(754-473))
ramp_Braeburn

# Use automated algorithm to identify p/q parameters
# Specify first difference = 1 and seasonal difference = 1
model1_Braeburn <- auto.arima(Braeburn.ts, seasonal=TRUE, xreg=cbind(step_Braeburn,ramp_Braeburn), max.d=1, max.D=1, stepwise=FALSE, trace=TRUE)
## Best model: Regression with ARIMA(1,0,1)(0,1,1)[52] errors 

# Check residuals
checkresiduals(model1_Braeburn)
Box.test(model1_Braeburn$residuals, lag = 24, type = "Ljung-Box")

# Estimate parameters and confidence intervals
summary(model1_Braeburn)
confint(model1_Braeburn)
## significance level
options(scipen = 999)
(1-pnorm(abs(model1_Braeburn$coef)/sqrt(diag(model1_Braeburn$var.coef))))*2
## autocorrelated residuals

# To forecast the counterfactual, model data excluding post-intervention time period
model2_Braeburn <- Arima(window(Braeburn.ts, end=decimal_date(as.Date('2017-10-09'))), order=c(1,0,1), seasonal=list(order=c(0,1,1), period=365.25/7))

# Forecast 12 months post-intervention and convert to time series object
fc_Braeburn <- forecast(model2_Braeburn, h=280)
fc_Braeburn.values=as.numeric(fc_Braeburn$mean)
fc_Braeburn.values = ifelse(fc_Braeburn.values>0,fc_Braeburn.values,0)
fc.ts_Braeburn <- ts(fc_Braeburn.values, start=time(Braeburn.ts)[length(Braeburn.ts)-280], frequency=365.25/7)

# Combine with observed data
Braeburn.ts.2 <- ts.union(Braeburn.ts, fc.ts_Braeburn)

# Plot
plot(Braeburn.ts.2, type="l", plot.type="s", col=c('blue','red'), xlab="Month", ylab="Dispensings", linetype=c("solid","dashed"), ylim=c(0,300))
abline(v=time(Braeburn.ts)[length(Braeburn.ts)-280], lty="dashed", col="gray")

Braeburn.ts_values = as.numeric(Braeburn.ts)
fc.ts_Braeburn_values = as.numeric(fc.ts_Braeburn)

# Define the frequency (52 for weekly data) and the start date
frequency <- 52
start_date <- as.Date("2008-09-22")
start_date2 <- as.Date("2017-10-09")

# Convert the numeric array to a weekly time series object
my_ts1 <- ts(Braeburn.ts_values, frequency = frequency, start = c(year(start_date), week(start_date)))
my_ts2 <- ts(fc.ts_Braeburn_values, frequency = frequency, start = c(year(start_date2), week(start_date)))
my_ts.2 <- ts.union(my_ts1, my_ts2)
plot(my_ts.2, type="l", plot.type="s", col=c('blue','red'), xlab="Month", ylab="Dispensings", linetype=c("solid","dashed"), ylim=c(0,200))
abline(v=weeks_from_start_Braeburn, lty="dashed", col="gray")
title("Braeburn shipments, 2008-2023")
