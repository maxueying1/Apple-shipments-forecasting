# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 23:35:36 2023

@author: m_xue
"""

import pandas as pd
import numpy as np
from pandas import DataFrame

## plot
import matplotlib.pyplot as plt
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_pacf

## Decomposition
import statsmodels.api as sm
from matplotlib import rcParams

## stationary -
## ADF test
from statsmodels.tsa.stattools import adfuller
## kpss test
from statsmodels.tsa.stattools import kpss

#####################################################################
# Models
#####################################################################
## ARIMA
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm

##Performance Measure
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt

##Exponential smoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing

##xgboost
from pandas import concat
from numpy import asarray
from xgboost import XGBRegressor

## FB Prophet
from prophet import Prophet
from pandas import to_datetime
import datetime as dt

## SARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from causalimpact import CausalImpact


'''
#####################################################################
# change work directory to script path
#####################################################################
import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
'''

## Read data
df = pd.read_csv('data.csv',index_col="date")
## remove the first 165 observations, so the data starts from 8/24/2015
#df = df.drop(index=range(365))
df.index = pd.to_datetime(df.index)

#####################################################################
# missing values
#####################################################################
# count the missing values in each column
missing_values_count = df.isnull().sum()
# print the count of missing values for each column
print("Missing values count for each column:\n", missing_values_count)
## should have less than 5% missing values => 38 missing values
##fullfill the missing values using linear interpolation
df=df.interpolate()
df.plot(linewidth=2,figsize=(16, 9))
'''
# Save the interplated DataFrames to Excel file
with pd.ExcelWriter('data-interppolated.xlsx') as writer:
    df.to_excel(writer, sheet_name='Sheet1', index=False)
'''

Date = df.index
Gala = df.Gala
HoneyCrisp = df.HoneyCrisp
CrippsPink = df.CrippsPink
RedDelicious = df.RedDelicious
Fuji = df.Fuji
GoldenDelicious = df.GoldenDelicious
GrannySmith = df.GrannySmith
Braeburn = df.Braeburn
CosmicCrisp = df.CosmicCrisp

#####################################################################
# Visualization
#####################################################################
pd.DataFrame(df).plot(subplots=True, layout=(3, 3), linewidth=2)

#####################################################################
# Summary Statistics
#####################################################################
print(Gala.describe())


#####################################################################
# Stationarity
#####################################################################

# ADF test
## NULL: nonstationary
## If the p-value is below a chosen significance level (e.g., 0.05), 
## we reject the null hypothesis and conclude that the series is stationary.
test_results = adfuller(Gala.dropna(),autolag='AIC')
print(f"ADF test statistic Gala: {test_results[0]}")
print(f"p-value Gala: {test_results[1]}")
print("Critical thresholds Gala:")
for key, value in test_results[4].items():
    print(f"\t{key}: {value}")


# KPSS test
# Null: stationary
## If the p-value is below a chosen significance level (e.g., 0.05), 
## we reject the null hypothesis and conclude that the series is non-stationary.
kpss_stat, p_value, lags, critical_values = kpss(Gala.dropna())
print(f'KPSS statistic Gala: {kpss_stat}')
print(f'p-value Gala: {p_value}')


rcParams["figure.figsize"] = 10, 6

##plot ACF and PACF
## Autocorrelation
fig = autocorrelation_plot(Gala)
plt.title("ACF - Gala")
plt.xlabel("Lag at k")
plt.ylabel("Correlation Coefficient")
plt.show();

fig = autocorrelation_plot(Gala).set_xlim([0, 24])
plt.title("ACF - Gala - 24 lags")
plt.xlabel("Lag at k")
plt.ylabel("Correlation Coefficient")
plt.show();

## Partial autocorrelation
fig = plot_pacf(Gala, lags=70)
plt.title("PACF - Gala")
plt.xlabel("Lag at k")
plt.ylabel("Correlation Coefficient")
plt.show();

## decomposition of gala
decomposition_gala = sm.tsa.seasonal_decompose(Gala, model='additive')
plt.figure(figsize=(10, 8))
plt.subplot(411)
plt.plot(Gala, label='Gala')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(decomposition_gala.trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(decomposition_gala.seasonal, label='Seasonal')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(decomposition_gala.resid, label='Residual')
plt.legend(loc='best')
plt.tight_layout()
plt.show()


#-----------------------------------------------------------------------------#
#----------------------------------- ARIMA -----------------------------------#
#-----------------------------------------------------------------------------#

split_point = len(Gala) - 52
train_Gala, test_Gala = Gala[0:split_point], Gala[split_point:]


#Automatically configure the ARIMA based on AIC
arima_auto = pm.auto_arima(train_Gala, 
                      start_p=0, 
                      start_q=0,
                      test='adf', # use adftest to find optimal 'd'
                      max_p=12, max_q=12, # maximum p and q
                      m=1, # frequency of series (if m==1, seasonal is set to FALSE automatically)
                      d=None,# let model determine 'd'
                      seasonal=False, # No Seasonality for standard ARIMA
                      trace=False, #logs 
                      error_action='warn', #shows errors ('ignore' silences these)
                      suppress_warnings=True,
                      information_criterion='aic',
                      stepwise=True)
print(arima_auto.summary())
## the best is found to be (1,0,2)

#### long term prediction
arima = ARIMA(train_Gala, order=(1,0,2))
arima_fit = arima.fit()
predictions_Gala_arima=arima_fit.forecast(52)
predictions_Gala_arima=predictions_Gala_arima.where(predictions_Gala_arima>0,0)
# evaluate forecasts
rmse = sqrt(mean_squared_error(test_Gala, predictions_Gala_arima))
mae = mean_absolute_error(test_Gala, predictions_Gala_arima)
print('Test RMSE: %.3f' % rmse)
print('Test MAE: %.3f' % mae)


## plot forecasts against actual outcomes, whole data
predictions_Gala2_arima = pd.DataFrame({"Gala":predictions_Gala_arima},index = Gala.index[-52:])
pyplot.plot(Gala[-52:], label='Actual Gala',color='blue')
pyplot.plot(predictions_Gala2_arima.Gala, label='Predicted - ARIMA', color='red')
plt.legend()
pyplot.show()


#-----------------------------------------------------------------------------#
#---------------------------------- SARIMA -----------------------------------#
#-----------------------------------------------------------------------------#

## The best model was found in R using auto.arima
## can also use auto.arima in python but it couldn't be ran on my pc

## The best SARIMA model is ARIMA(0,1,3)(2,1,0)[52] 
sarima_order_Gala = (0,1,3)  # Non-seasonal order
sarima_seasonal_order_Gala = (2,1,0,52)  # Seasonal order and seasonal period

# Fit the SARIMA model
sarima = SARIMAX(train_Gala, order=sarima_order_Gala, seasonal_order=sarima_seasonal_order_Gala)

#### long term prediction
sarima_fit = sarima.fit()
predictions_Gala_sarima=sarima_fit.forecast(52)
predictions_Gala_sarima=predictions_Gala_sarima.where(predictions_Gala_sarima>0,0)
# evaluate forecasts
rmse = sqrt(mean_squared_error(test_Gala, predictions_Gala_sarima))
mae = mean_absolute_error(test_Gala, predictions_Gala_sarima)
print('Test RMSE: %.3f' % rmse)
print('Test MAE: %.3f' % mae)


## plot forecasts against actual outcomes, whole data
predictions_Gala2_sarima = pd.DataFrame({"Gala":predictions_Gala_sarima},index = Gala.index[-52:])
pyplot.plot(Gala[-52:], label='Actual Gala',color='blue')
pyplot.plot(predictions_Gala2_sarima.Gala, label='Predicted - SARIMA', color='red')
plt.legend()
pyplot.show()

#-----------------------------------------------------------------------------#
#--------------------------- EXPONENTIAL SMOOTHING ---------------------------#
#-----------------------------------------------------------------------------#

split_point = len(Gala) - 52
train_Gala, test_Gala = Gala[0:split_point], Gala[split_point:]

##############################################################
############## Simple exponential smoothing ##################
##############################################################

ses = SimpleExpSmoothing(train_Gala, initialization_method="estimated")
## allow statsmodels to automatically find an optimized alpha
ses_fit= ses.fit()
ses_yhat=ses_fit.predict()
print('optimal alpha is %f' % ses_fit.params["smoothing_level"])

#### long term prediction
##optimal alpha is 0.995
## Performance checking
ses_fit = SimpleExpSmoothing(train_Gala).fit(smoothing_level=0.995,optimized=False)
## allow statsmodels to automatically find an optimized alpha
#ses_fit= ses.fit(smoothing_level=0.995,optimized=False)
predictions_Gala_ses=ses_fit.forecast(52)
predictions_Gala_ses=predictions_Gala_ses.where(predictions_Gala_ses>0,0)
# evaluate forecasts
rmse = sqrt(mean_squared_error(test_Gala, predictions_Gala_ses))
mae = mean_absolute_error(test_Gala, predictions_Gala_ses)
print('Test RMSE: %.3f' % rmse)
print('Test MAE: %.3f' % mae)


## plot forecasts against actual outcomes, whole data
predictions_Gala2_ses = pd.DataFrame({"Gala":predictions_Gala_ses},index = Gala.index[-52:])
pyplot.plot(Gala[-52:])
pyplot.plot(predictions_Gala2_ses.Gala, color='red')
pyplot.show()

# ----------------------------------------------------------------------------

##############################################################
############## Double/Triple exponential smoothing ###########
##############################################################

# ----------------------------------------------------------------------------
def holt_win_sea(y,y_to_train,y_to_test,seasonal_type,seasonal_period,predict_date):
    
    y.plot(marker='o', color='black', legend=True, figsize=(14, 7))
    
    if seasonal_type == 'additive':
        fit1 = ExponentialSmoothing(y_to_train, seasonal_periods = seasonal_period, trend='add', seasonal='add').fit()
        fcast1 = fit1.forecast(predict_date).rename('Additive')
        mse1 = ((fcast1 - y_to_test) ** 2).mean()
        print('The Root Mean Squared Error of additive trend, additive seasonal of '+ 
              'period season_length={} and a Box-Cox transformation {}'.format(seasonal_period,round(np.sqrt(mse1), 2)))
        
        fit2 = ExponentialSmoothing(y_to_train, seasonal_periods = seasonal_period, trend='add', seasonal='add', damped=True).fit()
        fcast2 = fit2.forecast(predict_date).rename('Additive+damped')
        mse2 = ((fcast2 - y_to_test) ** 2).mean()
        print('The Root Mean Squared Error of additive damped trend, additive seasonal of '+ 
              'period season_length={} and a Box-Cox transformation {}'.format(seasonal_period,round(np.sqrt(mse2), 2)))
        
        fit1.fittedvalues.plot(style='--', color='red')
        fcast1.plot(style='--', marker='o', color='red', legend=True)
        fit2.fittedvalues.plot(style='--', color='green')
        fcast2.plot(style='--', marker='o', color='green', legend=True)

    
    elif seasonal_type == 'multiplicative':  
        fit3 = ExponentialSmoothing(y_to_train, seasonal_periods = seasonal_period, trend='add', seasonal='mul').fit()
        fcast3 = fit3.forecast(predict_date).rename('Multiplicative')
        mse3 = ((fcast3 - y_to_test) ** 2).mean()
        print('The Root Mean Squared Error of additive trend, multiplicative seasonal of '+ 
              'period season_length={} and a Box-Cox transformation {}'.format(seasonal_period,round(np.sqrt(mse3), 2)))
        
        fit4 = ExponentialSmoothing(y_to_train, seasonal_periods = seasonal_period, trend='add', seasonal='mul', damped=True).fit()
        fcast4 = fit4.forecast(predict_date).rename('Multiplicative+damped')
        mse4 = ((fcast3 - y_to_test) ** 2).mean()
        print('The Root Mean Squared Error of additive damped trend, multiplicative seasonal of '+ 
              'period season_length={} and a Box-Cox transformation {}'.format(seasonal_period,round(np.sqrt(mse4), 2)))
        
        fit3.fittedvalues.plot(style='--', color='red')
        fcast3.plot(style='--', marker='o', color='red', legend=True)
        fit4.fittedvalues.plot(style='--', color='green')
        fcast4.plot(style='--', marker='o', color='green', legend=True)
        
    else:
        print('Wrong Seasonal Type. Please choose between additive and multiplicative')

    plt.show()
    
holt_win_sea(Gala, train_Gala , test_Gala,'multiplicative',52, 52)


#### long term prediction
fit2 = ExponentialSmoothing(train_Gala, seasonal_periods = 52, trend='add', seasonal='add', damped=True).fit()
predictions_Gala_hwes=fit2.forecast(52)
predictions_Gala_hwes=predictions_Gala_hwes.where(predictions_Gala_hwes>0,0)
# evaluate forecasts
rmse = sqrt(mean_squared_error(test_Gala, predictions_Gala_hwes))
mae = mean_absolute_error(test_Gala, predictions_Gala_hwes)
print('Test RMSE: %.3f' % rmse)
print('Test MAE: %.3f' % mae)


## plot forecasts against actual outcomes, whole data
predictions_Gala2_hwes = pd.DataFrame({"Gala":predictions_Gala_hwes},index = Gala.index[-52:])
pyplot.plot(Gala[-52:])
pyplot.plot(predictions_Gala2_hwes.Gala, color='red')
pyplot.show()


# ----------------------------------------------------------------------------


#-----------------------------------------------------------------------------#
#---------------------------------- XGBOOST ----------------------------------#
#-----------------------------------------------------------------------------#


# ----------------------------------------------------------------------------
## below is from machinelearningmastery xgboost
# transform a time series dataset into a supervised learning dataset
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[0]
	df = DataFrame(data)
	cols = list()
	# input sequence (t-n, ... t-1)
    ## the number of lags
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
	# put it all together
	agg = concat(cols, axis=1)
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg.values
##idea is to use the # of n_in lags to predict # of n_out values

# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
	return data[:-n_test, :], data[-n_test:, :]

# fit an xgboost model and make a one step prediction
def xgboost_forecast(train, testX):
	# transform list into array
	train = asarray(train)
	# split into input and output columns
	trainX, trainy = train[:, :-1], train[:, -1]
	# fit model
	model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
	model.fit(trainX, trainy)
	# make a one-step prediction
	yhat = model.predict(asarray([testX]))
	return yhat[0]

# walk-forward validation for univariate data
def walk_forward_validation(data, n_test):
	predictions = list()
	# split dataset
	train, test = train_test_split(data, n_test)
	# seed history with training dataset
	history = [x for x in train]
	# step over each time-step in the test set
	for i in range(len(test)):
		# split test row into input and output columns
		testX, testy = test[i, :-1], test[i, -1]
		# fit model on history and make a prediction
		yhat = xgboost_forecast(history, testX)
		# store forecast in list of predictions
		predictions.append(yhat)
		# add actual observation to history for the next loop
		history.append(test[i])
		# summarize progress
		print('>expected=%.1f, predicted=%.1f' % (testy, yhat))
	# estimate prediction error
	mae = mean_absolute_error(test[:, -1], predictions)
	return mae, test[:, 1], predictions

#######################
## choose the lag with lowest mae
d=[]
yhat_save=[]
for i in range(1,11):
    svm_gala = series_to_supervised(Gala, n_in=i)
    #svm_gala = svm_gala[:,1::2]
    mae, y, yhat = walk_forward_validation(svm_gala, 52)
    yhat_save.append(yhat)
    d.append(mae)
print(d.index(min(d))+1)
lag_gala=d.index(min(d))+1
##the optimal lag is 4


#### long term prediction
## Performance checking
history = [x for x in train_Gala]
predictions_Gala_xgboost = list()
for t in range(len(test_Gala)):
    svm_gala = series_to_supervised(history, 4)    
    output = xgboost_forecast(svm_gala, svm_gala[-1,1:])
    yhat = output
    predictions_Gala_xgboost.append(yhat)
    obs = test_Gala[t]
    history.append(output)
    print('predicted=%f, expected=%f' % (yhat, obs))
# evaluate forecasts
predictions_Gala_xgboost = [0 if x < 0 else x for x in predictions_Gala_xgboost]
rmse = sqrt(mean_squared_error(test_Gala, predictions_Gala_xgboost))
mae = mean_absolute_error(test_Gala, predictions_Gala_xgboost)
print('Test RMSE: %.3f' % rmse)
print('Test MAE: %.3f' % mae)

## plot forecasts against actual outcomes, whole data
predictions_Gala2_xgboost = pd.DataFrame({"Gala":predictions_Gala_xgboost},index = Gala.index[-52:])
pyplot.plot(Gala[-52:])
pyplot.plot(predictions_Gala2_xgboost.Gala, color='red')
pyplot.show()


# ----------------------------------------------------------------------------


#-----------------------------------------------------------------------------#
#---------------------------------- PROPHET ----------------------------------#
#-----------------------------------------------------------------------------#

##The DataFrame must have a specific format. 
##The first column must have the name ‘ds‘ and contain the date-times. 
##The second column must have the name ‘y‘ and contain the observations.

##prepare expected column names
df_Gala=pd.DataFrame({"y":Gala.values, "ds": Gala.index})
split_point = len(df_Gala) - 52
train_df_Gala, test_df_Gala = df_Gala[0:split_point], df_Gala[split_point:]

# ----------------------------------------------------------------------------
######### automatic changepoint detection

## changepoint_range
##By default, Prophet specifies 25 potential changepoints 
##which are uniformly placed in the first 80% of the time series.

## changepoint_prior_scale
## By default, this parameter is set to 0.05. 
## Increasing it will make the trend more flexible
# define the model
prophet = Prophet(interval_width=0.95, changepoint_range=0.9, 
                  changepoint_prior_scale = 0.5)
## DEFAULT CHANGEPOINTG RANGE IS 80%
# fit the model
prophet.fit(train_df_Gala)
# define the period for which we want a prediction
prediction = list()
initial_date = df_Gala['ds'].max()- dt.timedelta(weeks = 52)
for i in range(1, 53):
	date = initial_date + dt.timedelta(weeks = i)
	prediction.append([date])
prediction = DataFrame(prediction)
prediction.columns = ['ds']
prediction['ds']= to_datetime(prediction['ds'])

# use the model to make a forecast
forecast = prophet.predict(prediction)

# summarize the forecast
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())
# plot forecast
prophet.plot(forecast,uncertainty=True)
pyplot.scatter(Gala.iloc[-52:].index, Gala.iloc[-52:].values, s=8,color='black')
pyplot.show()

predictions_Gala_prophet=forecast['yhat']
predictions_Gala_prophet=predictions_Gala_prophet.where(predictions_Gala_prophet>0,0)
# evaluate forecasts
rmse = sqrt(mean_squared_error(test_Gala, predictions_Gala_prophet.values))
mae = mean_absolute_error(test_Gala, predictions_Gala_prophet)
print('Test RMSE: %.3f' % rmse)
print('Test MAE: %.3f' % mae)


## plot forecasts against actual outcomes, whole data
predictions_Gala2_prophet = pd.DataFrame({"Gala":predictions_Gala_prophet.values},index = Gala.index[-52:])
pyplot.plot(Gala[-52:])
pyplot.plot(predictions_Gala2_prophet.Gala, color='red')
pyplot.show()

# ----------------------------------------------------------------------------


## plot forecasts against actual outcomes, whole data
pyplot.plot(Gala[-52:],label='actual Gala',color='blue')
pyplot.plot(predictions_Gala2_arima.Gala,label='predicted - ARIMA',color='olive')
pyplot.plot(predictions_Gala2_sarima.Gala,label='predicted - SARIMA',color='magenta')
pyplot.plot(predictions_Gala2_ses.Gala,label='predicted - SES',color='orange')
pyplot.plot(predictions_Gala2_hwes.Gala,label='predicted - HWES',color='green')
pyplot.plot(predictions_Gala2_xgboost.Gala,label='predicted - XGBOOST',color='red')
pyplot.plot(predictions_Gala2_prophet.Gala,label='predicted - RPOPHET',color='purple')
plt.legend()
plt.title('Gala')
pyplot.show()

#-----------------------------------------------------------------------------#
#---------------------------- CAUSAL EFFECT ----------------------------------#
#-----------------------------------------------------------------------------#


######################################
##### Appearance of Cosmic crisp #####
######################################

# define the 'before' and 'after' periods

# 2019-11-25  2019-12-02
## 2017-08-14 2017-08-21

## start date
pre_period  = [ pd.Timestamp('2008-08-25') , pd.Timestamp('2017-08-14') ]
post_period = [ pd.Timestamp('2017-08-21') , pd.Timestamp('2023-02-27') ]

impact = CausalImpact(Gala, 
                  pre_period, post_period, 
                  nseasons=[{'period': 12}],
                  prior_level_sd=0.05)
# print out a summary
print(impact.summary())
impact.plot()
# display the plots
impact.plot(panels=['pointwise'], figsize=(12, 4));

print(impact.summary('report'))

