'''
SOC 423 - Birth Rate Prediction Model
Author: Yuya Ong
'''
from __future__ import print_function
import os
import sys
import numpy as np
import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Load Dataset
birth_rate = pd.read_csv('../data/raw/birth_rate.csv', index_col=1)
birth_rate = birth_rate.drop(['Country Name', 'Indicator Name', 'Indicator Code', '2017', 'Unnamed: 62'], axis=1)
birth_rate = birth_rate.dropna().T

# MAPE Output
err = open('../log/ARIMA_MAPE.csv', 'w')

# Setup Data Structure
for country in list(birth_rate):
    data = birth_rate[country].to_frame()
    data = data.rename(columns={country:'y'})
    data['ds'] = pd.Series(map(lambda x: str(x) + '-01-01', list(data.T)), index=data.index)

    # Swap Column Order
    cols = data.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    data = data[cols]

    # Train Model
    m = Prophet(yearly_seasonality=False, daily_seasonality=False, weekly_seasonality=False)
    m.fit(data[:-20])

    # Generate Prediction
    future = m.make_future_dataframe(periods=20, freq='Y')
    forecast = m.predict(future)

    # Compute Error
    error = mape(data['y'].values, forecast['trend'].values)
    err.write(country+','+str(error)+'\n')

    # Generate Plots
    print(forecast.head())
    out = np.array([forecast['trend'].tolist(), data['y'].tolist()])
    plt.plot(out.T)
    plt.xticks(plt.xticks()[0], list(range(1950, 2016, 10)))

    plt.suptitle('Birth Rate Projection ['+country+'] (ARIMA)')
    plt.title('MAPE: ' + str("{0:.2f}".format(error)) + '%')
    plt.legend(['Predicted', 'Actual'])
    plt.xlabel('Year')
    plt.ylabel('Birth Rate (Per Woman)')

    plt.savefig('../plot/ARIMA/'+country+'.png')
    plt.clf()

err.close()
