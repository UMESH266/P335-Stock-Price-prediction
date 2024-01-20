# Streamlit code development for the project code P335 and name Prediction of stock price
import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns

# Navigation bar
check = st.sidebar.radio("My Project", ('About', 'Team', 'ML - Modeling'))
if check == 'About':
    # Page title
    st.title('Stock price prediction')

    # Image
    st.image('./stock_price.jpeg')
    st.write('-------')

    st.subheader('1. Introduction')
    st.markdown('#### 1.1 Project program of ExcelR')
    st.text(""" 
    ExcelR provides opportunity to work on Real life data based, 2 projects in its Data Science Training 
    program. Projects allotment and commencement starts after submission of mandatory assignments related 
    to the course within specified time. These projects are mentor guided, team work. Also, there will be
    regular review meetings and deadlines for each phase of project.
    """)
    st.write('-------')

    st.markdown('#### 1.2  Objectives')
    st.text("""
    i) Exposure to Real life problems
    ii) Team work
    iii) Application of classroom training
    """)
    st.write('-------')

    st.markdown('#### 1.3  Project details')
    st.text("""
    Project Name        :  Stock Price Prediction
    Project code        :  P335
    Project Mentor      :  Mrs Snehal Shinde
    Project coordinator :  Mrs S Mounika
    Kick-off-Date       :  05-01-2024
    """)
    st.write('-------')

if check == 'Team':
    st.title('My TEAM')
    st.image('./team_work_image.jpg')
    st.subheader('Team details')
    st.text("""
    Group number     :   05
    
    Group Members    :   07
    
    Names of Team members:
    
    1. Mr. Abhijit Chandrakant Raut
    
    2. Ms. Anjali Awadhesh Pal
    
    3. Mohammad Riza Amanathussain Sanadi
    
    4. Ms. Samiksha Shirish Hate
    
    5. Mr. Tumma Shivkumar Ashokrao
    
    6. Mr. Umesh
    
    7. Ms. Vidya Surbhi
    """)

if check == 'ML - Modeling':
    st.subheader("Tesla stock price prediction")
    st.write('---')
    section = st.sidebar.radio('Model building', ['Know your data', 'Clean the Data', 'Data Stats',
                                                  'Graph the data', 'ARIMA model', 'Visualization of Results'])
    global data1, data, result
    # Loading raw dataset
    data1 = pd.read_csv("./Tesla_raw.csv")

    if section == 'Know your data':
        st.markdown('#### Know your DATA')
        # Display of dataset
        st.write('Dataset of Tesla  having stock details for last 10 years.')
        st.write('Size of dataset in rows and columns: ', data1.shape)
        st.dataframe(data1)
        st.write('----')
        st.write('Top five records:')
        st.write(data1.head(5))
        st.write('Bottom five records:')
        st.write(data1.tail(5))
        st.write('---')

        st.write('Data types of columns in dataset:')
        st.dataframe(data1.dtypes)
        st.write('----')

        st.markdown('#### Inferences drawn:')
        st.text("""
        1. Tesla stock prices downloaded from official website of NASDAQ.
        2. Columns present in the dataset are Date, Close/Last, Volume, Open, 
           High and Low. 
        3. Recent 10 years stock prices data.
        4. Time format of Date column is MM/DD/YYYY and needs to be converted to 
           datetime format.
        5. Stock prices are in $ (Dollars).
        6. Close/Last column needs to renamed to Close for convenience.
        7. Currency data values decimals are inconsistent and needs to be standardized.
        8. Dtypes of price features needs to be converted.
        """)

    # Cleaning data
    @st.cache_data
    def dataset_cleaner(dataset):
        # Dropping of NaN values
        dataset.dropna(inplace=True)

        # Renaming column Close / Last
        dataset.rename(columns={'Close/Last': 'Close'}, inplace=True)

        # Conversion of Dtypes of columns
        # Replacing special character '/' and replacing with '-' of data column
        dataset['Date'] = dataset['Date'].apply(lambda x: x.replace('/', '-'))

        # Date column from string to datetime format
        dataset['Date'] = pd.to_datetime(dataset['Date'])

        # Close, Open, High and low from string to float type
        for column in ['Close', 'Open', 'High', 'Low']:
            # Removal of special character '$' from the data values
            dataset[column] = dataset[column].apply(lambda x: x.replace('$', ''))

            # Conversion of dtype string to float
            dataset[column] = dataset[column].astype(float)

            # Rounding off data values to two decimal places
            dataset[column] = dataset[column].apply(lambda x: np.round(x, 2))
        return dataset

    if section == 'Clean the Data':
        st.markdown('#### Cleaned data')
        st.write('Dataset')
        cleaned_data = dataset_cleaner(data1)
        st.write(cleaned_data)

        st.write('Consistent Data types')
        st.write(cleaned_data.dtypes)
        cleaned_data = cleaned_data.set_index('Date')
        cleaned_data.to_csv('./clean_data.csv')

    data = pd.read_csv(r'./clean_data.csv', index_col='Date')
    model_data = data['2023-12-29':]['Close'].sort_index()
    actual_data = data[:]['Close'].sort_index()
    actual_data = actual_data.reset_index()
    actual_data = actual_data.iloc[2505:]

    if section == 'Data Stats':
        st.markdown('#### Descriptive statistics: ')
        data = pd.read_csv(r'./clean_data.csv')
        st.write(data.describe())
        st.write('---')

        st.markdown('#### Observations made')
        st.text("""
        Statistics of price features such as Close, Open, High, Low are close to each 
        other and shows that their nature will be similar in every aspect.
        
        Whereas, Volume has has more noise confirming more variation in volume of stock 
        trading.
        
        High range can be observed in all the columns.
        """)

    if section == 'Graph the data':
        st.markdown("#### Visualization of dataset")
        st.write('Dataset distribution')

        fig = sns.pairplot(data, kind='scatter', diag_kind='kde')
        st.pyplot(fig)

        st.markdown('Chart of Close price over time')
        st.line_chart(model_data, y='Close')
        st.write('-------')

        st.markdown('#### Observations: ')
        st.text("""
        Dataset:
        1. Kernel density (kde) plots helps to visualize the distribution of values in the 
        time series. Also, helps in identifying outliers and understanding the data's 
        central tendencies.
        
        2. It can be observed from above pair plot that distribution of columns is not normal 
        and all features are right skewed (+ve skewness).
        
        3. Volume feature has high kurtosis value and confirms presence of outliers. Whereas, 
        other features have low kurtosis and have low / nil outliers.
        
        Close price:
        1. Line plot of Close price shows uptrend and no observable patterns.
        2. Raise in price of stock is more rigorous between 2020 to 2022.
        3. Significant fall in 2023 can be observed.
        4. Overall price of stock shows uptrend.
        """)

    # Model building
    arima = SARIMAX(model_data, order=(1, 0, 0), seasonal_order=(2, 1, 0, 12))
    result = arima.fit()
    test = model_data['2022-01-03':]
    if section == 'ARIMA model':
        model_fit = 1
        st.subheader("ARIMA - time series prediction model")
        summary = result.summary()
        st.write('##### Model Summary: ')
        st.write(summary)
        st.write('----')

        # Evaluation of model
        # Predictions
        predictions = result.get_prediction(start='2022-01-03')
        preds = predictions.predicted_mean

        st.write('##### Model Performance')
        # Calculate mean squared error

        MSE = np.round(mean_squared_error(test, preds), 2)
        st.write('MSE value : ', MSE)

        # Calculate root mean squared error
        RMSE = np.round(np.sqrt(MSE), 2)
        st.write('RMSE value : ', RMSE)

        # Mean Absolute Error
        MAE = mean_absolute_error(test, preds)
        st.write('MAE value  : ', np.round(MAE, 2))
        st.write('-------')

    if section == 'Visualization of Results':
        st.subheader('Visualization of Results')
        plot = st.sidebar.radio('Select plot', ['Predicted prices on test set', 'Forecasted prices',
                                                'Comparison'])
        if plot == 'Predicted prices on test set':
            # Predictions
            # Line chart of predicted Close prises
            st.text('Predicted Close prises')
            pred = result.get_prediction(start='2022-01-03', dynamic=False)
            pred_ci = pred.conf_int()

            chart_data = {'Predicted price': pred.predicted_mean,
                          "Lower bound": pred_ci.iloc[:, 0],
                          "Upper bound": pred_ci.iloc[:, 1],
                          "Actual price": test
                          }

            st.line_chart(pd.DataFrame(chart_data))

        if plot == 'Forecasted prices':
            # Visualization of Forecasting for next 30 steps
            st.write('Forecasted Close price')
            step = st.slider('Select steps for forecast', min_value=1, max_value=365, step=30, value=30)
            pred_uc = result.get_forecast(steps=step)
            pred_uc_ci = pred_uc.conf_int()
            forecast_data = {'Forecasted price': pred_uc.predicted_mean,
                             "Lower bound": pred_uc_ci.iloc[:, 0],
                             "Upper bound": pred_uc_ci.iloc[:, 1]
                             }
            st.line_chart(pd.DataFrame(forecast_data))

        if plot == 'Comparison':
            st.write('Actual vs Forecasted Close prices')
            pred_uc = result.get_forecast(steps=12)
            pred_uc_ci = pred_uc.conf_int()
            forecast_data = {'Forecasted price': pred_uc.predicted_mean,
                             "Lower bound": pred_uc_ci.iloc[:, 0],
                             "Upper bound": pred_uc_ci.iloc[:, 1],
                             "Actual price": actual_data['Close']
                             }
            # st.write(pred_uc.predicted_mean)
            # st.write(actual_data)
            st.line_chart(pd.DataFrame(forecast_data))
            st.subheader('Thank you . . ')
