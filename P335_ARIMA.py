# Streamlit code development for the project code P335 and name Prediction of stock price
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, r2_score

# Page title
st.title('Stock price prediction - Tesla')

# Image
st.image('./stock_price.jpeg')

# Loading dataset
data = pd.read_csv("./Tesla.csv", index_col='Date')

# Display of data
st.write("### View of Tesla stock Close price")
st.text('Clean Dataset of Tesla having time series Close price of last 10 years.')
st.write('Size of dataset in rows and columns: ', data.shape)
st.dataframe(data)

# Model building
st.write("### ARIMA - time series prediction model")
st.write('#### Model Summary: ')
arima = SARIMAX(data, order=(1, 0, 0), seasonal_order=(2, 1, 0, 12))

result = arima.fit()
summary = result.summary()
st.write(summary)

# Evaluation of model
# Predictions
predictions = result.get_prediction(start='2022-01-03')
preds = predictions.predicted_mean

st.write('#### Model Performance')
# Calculate mean squared error
test = data['2022-01-03':]
MSE = mean_squared_error(test, preds)
st.write('MSE value: ', MSE)

# Calculate root mean squared error
RMSE = np.sqrt(MSE)
st.write('RMSE value: ', RMSE)

# Goodness of fit of model
fit = r2_score(test, preds)
st.write('Goodness of fit: ', np.round(fit * 100, 2), '%')

# Visualization of predictions
st.write('#### Visualization')
st.text('Line chart of Close price')
st.line_chart(data, y='Close')

# Predictions
# Line chart of predicted Close prises
st.write('Line chart of predicted Close prises')
pred = result.get_prediction(start='2022-01-03', dynamic=False)
pred_ci = pred.conf_int()
st.line_chart(pred.predicted_mean)
# st.map(pred_ci)
# ax.fill_between(pred_ci.index,
#                 pred_ci.iloc[:, 0],
#                 pred_ci.iloc[:, 1], color='k', alpha=.2)
# ax.set_xlabel('Date')
# ax.set_ylabel('Close price')
# ax.set_xlim(xmin='2020')
# plt.legend()
# plt.show()
#
# # Visualization of Forecasting for next 30 steps
# pred_uc = result_1.get_forecast(steps=30)
# pred_ci = pred_uc.conf_int()
# pred_uc.predicted_mean.plot(label='Forecast')
# plt.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='k', alpha=.7)
# plt.xlabel('Date')
# plt.ylabel('Close price')
# plt.legend(loc='best')
# plt.title('Forecasting of Close price of stock Tesla with confidence interval')
# # plt.xlim(xmin='2014')
# plt.show()
