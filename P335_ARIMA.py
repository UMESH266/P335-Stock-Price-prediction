# Streamlit code development for the project code P335 and name Prediction of stock price
import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Page title
st.title('Stock price prediction - Tesla')

# Image
st.image('./stock_price.jpeg')
st.write('-------')

# Navigation bar
check = st.sidebar.radio("Navigation", ('About', 'Team', 'Project'))
if check == 'About':
    st.subheader('1.1  Project program of ExcelR')
    st.write(""" 
    ExcelR provides opportunity to work on Real life data based, 2 projects in its Data Science Training 
    program. Projects allotment and commencement starts after submission of mandatory assignments related 
    to the course within specified time. These projects are mentor guided, team work. Also, there will be
    regular review meetings and deadlines for each phase of project.
    """)
    st.write('-------')
    st.subheader('1.2  Objectives')
    st.text("""
    i) Exposure to Real life problems
    ii) Team work
    iii) Application of classroom training
    """)
    st.write('-------')
    st.subheader('1.3  Project details')
    st.text("""
    Project Name        :  Stock Price Prediction
    Project code        :  P335
    Project Mentor      :  Mrs Snehal Shinde
    Project coordinator :  S Mounika
    Kick-off-Date       :  05-01-2024
    """)
    st.write('-------')
if check == 'Team':
    st.subheader('Team details')
    st.text("""
    Group number     :   05
    Group Members    :   07
    Names of Team members:
    1. Mr. Abhijit Chandrakant Raut
    2. MISS. ANJALI AWADHESH PAL
    3. Mohammad Riza Amanathussain Sanadi
    4. Ms. Samiksha Shirish Hate
    5. Mr. Tumma Shivkumar Ashokrao
    6. Mr. Umesh
    7. Ms. Vidya Surbhi
    """)

if check == 'Project':
    # Loading raw dataset
    data1 = pd.read_csv("./Tesla Inc.csv")
    # Cleaned data
    data = pd.read_csv("./Tesla.csv", index_col='Date')

    # Display of dataset
    st.subheader("Tesla stock")
    st.write('Dataset of Tesla  having stock details for last 10 years.')
    st.write('Size of dataset in rows and columns: ', data1.shape)
    st.dataframe(data1)
    st.write('---')
    # Model building
    st.subheader("ARIMA - time series prediction model")
    st.write('##### Model Summary: ')
    arima = SARIMAX(data, order=(1, 0, 0), seasonal_order=(2, 1, 0, 12))

    result = arima.fit()
    summary = result.summary()
    st.write(summary)

    # Evaluation of model
    # Predictions
    predictions = result.get_prediction(start='2022-01-03')
    preds = predictions.predicted_mean
    st.write('-------')
    st.write('##### Model Performance')
    # Calculate mean squared error
    test = data['2022-01-03':]
    MSE = np.round(mean_squared_error(test, preds), 2)
    st.write('MSE value : ', MSE)

    # Calculate root mean squared error
    RMSE = np.round(np.sqrt(MSE), 2)
    st.write('RMSE value : ', RMSE)

    # Goodness of fit of model
    MAE = mean_absolute_error(test, preds)
    st.write('MAE value  : ', np.round(MAE, 2))
    st.write('-------')

    # Visualization of predictions
    st.write('##### Visualization')
    st.text('Chart of Close price')
    st.line_chart(data, y='Close')
    st.write('-------')

    # Predictions
    # Line chart of predicted Close prises
    st.text('Predicted Close prises')
    pred = result.get_prediction(start='2022-01-03', dynamic=False)
    pred_ci = pred.conf_int()
    # st.line_chart(pred.predicted_mean)
    chart_data = {'Predicted_mean': pred.predicted_mean,
                  "Lower_bound": pred_ci.iloc[:, 0],
                  "Upper_bound": pred_ci.iloc[:, 1]
                  }

    st.line_chart(pd.DataFrame(chart_data))

    # Visualization of Forecasting for next 30 steps
    st.write('Forecasted Close price for next 30 steps')
    pred_uc = result.get_forecast(steps=30)
    pred_uc_ci = pred_uc.conf_int()
    forecast_data = {'Forecasted price': pred_uc.predicted_mean,
                  "Lower_bound": pred_uc_ci.iloc[:, 0],
                  "Upper_bound": pred_uc_ci.iloc[:, 1]
                  }

    st.line_chart(pd.DataFrame(forecast_data), )

    st.subheader('Thank you . . .')
