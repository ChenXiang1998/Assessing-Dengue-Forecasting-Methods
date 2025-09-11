import pandas as pd
from prophet import Prophet

def predict_prophet(data, window_size):
    prediction_table = pd.DataFrame()  # Initialize the dataframe to store results

    # Loop through the dataset in windows, skipping the last window_size + 12 weeks for out-of-bounds
    for i in range(10):
        # Prepare the data slice for training the model
        train_data = data.iloc[i:i + window_size].copy()
        train_data.columns = ['ds', 'y']  # Prophet expects column names 'ds' and 'y'

        # Initialize and fit the Prophet model
        model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False,
                        seasonality_prior_scale=15,# Adjust based on your data; higher values fit larger seasonal fluctuations
                        changepoint_prior_scale=0.1  # Adjust for more flexible trend changes
                        )
        model.fit(train_data)

        # Make a dataframe for future predictions
        future_dates = model.make_future_dataframe(periods=12, freq='W')  # 'W' for weekly frequency

        # Use the model to make predictions
        forecast = model.predict(future_dates)

        # Extract only the necessary future predictions
        future_forecast = forecast[['ds', 'yhat']].tail(12)

        # Prepare the output for this iteration
        predictions = {
            'Datetime': train_data.iloc[-1]['ds'],
            'Real_Cases': train_data.iloc[-1]['y'],
            **{f'Predicted_Cases_Week{n}': future_forecast.iloc[n-1]['yhat'] for n in [1, 2, 3, 4, 8, 12]}
        }

        # Append the predictions of this window to the prediction table
        prediction_table = pd.concat([prediction_table, pd.DataFrame([predictions])], ignore_index=True)

    return prediction_table

data = pd.read_csv('data_new.csv')
data['ds'] = pd.to_datetime(data['data_iniSE'])
data = data[['ds', 'casos']]  # Only keep necessary columns

window_size = 313  # Set window size to 313 weeks



results_df = predict_prophet(data, window_size)


print(results_df.head())

def lag_predictions(df):
    # Shift the prediction columns to introduce lags
    df['Predicted_Cases_Week2'] = df['Predicted_Cases_Week2'].shift(1)  # Lag by 1
    df['Predicted_Cases_Week3'] = df['Predicted_Cases_Week3'].shift(2)  # Lag by 2
    df['Predicted_Cases_Week4'] = df['Predicted_Cases_Week4'].shift(3)  # Lag by 3
    df['Predicted_Cases_Week8'] = df['Predicted_Cases_Week8'].shift(7)  # Lag by 7
    df['Predicted_Cases_Week12'] = df['Predicted_Cases_Week12'].shift(11) # Lag by 11

    return df

# Use the function
results_df = lag_predictions(results_df)
print(results_df.head(20))  # Print the first 20 rows to see both original and lagged values

results_df.to_csv('prop.csv', index=False)

def predict_prophet(data, window_size):
    prediction_table = pd.DataFrame()  # Initialize the dataframe to store results

    for i in range(len(data) - window_size):  # Adjust index range to avoid NaNs
        train_data = data.iloc[i:i + window_size].copy()
        train_data.rename(columns={'casos': 'y', 'datetime': 'ds'}, inplace=True)

        model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False,
                        seasonality_prior_scale=10, changepoint_prior_scale=0.5)
        model.add_regressor('tempmed')
        model.add_regressor('umidmed')

        model.fit(train_data)

        future_dates = model.make_future_dataframe(periods=12, freq='W')
        # Check if there is enough data for regressors; if not, repeat the last value
        last_available_idx = i + window_size - 1
        last_tempmed = data.loc[last_available_idx, 'tempmed']
        last_umidmed = data.loc[last_available_idx, 'umidmed']

        future_dates['tempmed'] = last_tempmed
        future_dates['umidmed'] = last_umidmed

        # If there is sufficient data available, use it
        if i + window_size + 12 <= len(data):
            tempmed_values = data['tempmed'].iloc[i + window_size:i + window_size + 12].reset_index(drop=True)
            umidmed_values = data['umidmed'].iloc[i + window_size:i + window_size + 12].reset_index(drop=True)
            future_dates.loc[:len(tempmed_values)-1, 'tempmed'] = tempmed_values
            future_dates.loc[:len(umidmed_values)-1, 'umidmed'] = umidmed_values

        forecast = model.predict(future_dates)
        future_forecast = forecast[['ds', 'yhat']].tail(12)

        predictions = {
            'Datetime': train_data.iloc[-1]['ds'],
            'Real_Cases': train_data.iloc[-1]['y'],
            **{f'Predicted_Cases_Week{n}': future_forecast.iloc[n-1]['yhat'] for n in [1, 2, 3, 4, 8, 12]}
        }
        prediction_table = pd.concat([prediction_table, pd.DataFrame([predictions])], ignore_index=True)

    return prediction_table

# Example usage
data = pd.read_csv('data.csv')
data['datetime'] = pd.to_datetime(data['data_iniSE'])  # Ensure datetime is the correct dtype
data = data[['datetime', 'casos', 'tempmed', 'umidmed']]  # Keep necessary columns with covariates

window_size = 313  # Set window size to 313 weeks
results_df = predict_prophet(data, window_size)
print(results_df.head())
