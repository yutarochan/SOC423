'''
SOC 423 - Birth Rate Prediction Model
Author: Yuya Ong
'''
from __future__ import print_function
import os
import sys
import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt

# Load Dataset
birth_rate = pd.read_csv('../data/raw/birth_rate.csv', index_col=1)
birth_rate = birth_rate.drop(['Country Name', 'Indicator Name', 'Indicator Code', '2017', 'Unnamed: 62'], axis=1)
birth_rate = birth_rate.dropna().T

# Setup Data Structure
data = birth_rate['RUS'].to_frame()
data = data.rename(columns={'RUS':'y'})
data['ds'] = pd.Series(map(lambda x: str(x) + '-01-01', list(data.T)), index=data.index)

# Swap Column Order
cols = data.columns.tolist()
cols = cols[-1:] + cols[:-1]
data = data[cols]

print(data.head())

# Train Model
m = Prophet(yearly_seasonality=False, daily_seasonality=False, weekly_seasonality=False)
m.fit(data)

future = m.make_future_dataframe(periods=25, freq='Y')
print(future.tail())

forecast = m.predict(future)
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

fig1 = m.plot(forecast)
plt.show(fig1)
