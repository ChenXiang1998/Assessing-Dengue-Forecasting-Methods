# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def predict_rf(data, window_size):
    # Initialize the results DataFrame
    prediction_table = pd.DataFrame(columns=['Datetime', 'Real_Cases', 'Predicted_Cases_Week1',
                                             'Predicted_Cases_Week2', 'Predicted_Cases_Week3',
                                             'Predicted_Cases_Week4', 'Predicted_Cases_Week8',
                                             'Predicted_Cases_Week12'])

    # Initialize the model
    rf_model = RandomForestRegressor(n_estimators=5000)

    # Start the moving-window prediction len(data) - window_size
    for i in range(len(data) - window_size):  # Adjusting to ensure we have enough data for the furthest prediction
        current_window = data.iloc[i:(i + window_size)]
        X_train = current_window.drop(columns=['data_iniSE', 'casos'])  # Assuming datetime and casos columns exist
        y_train = current_window['casos']

        # Fit the model
        rf_model.fit(X_train, y_train)

        # Dictionary to store predictions for different weeks ahead
        predictions = {}
        for weeks_ahead in [1, 2, 3, 4, 8, 12]:
            if i + window_size + weeks_ahead - 1 < len(data):
                X_test = data.iloc[i + window_size + weeks_ahead - 1:i + window_size + weeks_ahead].drop(columns=['data_iniSE', 'casos'])
                prediction = rf_model.predict(X_test)[0]
                predictions[f'Predicted_Cases_Week{weeks_ahead}'] = prediction

        # Create a new row and append it to the DataFrame
        new_row = {
            'Datetime': data.iloc[i + window_size]['data_iniSE'],
            'Real_Cases': data.iloc[i + window_size]['casos'],
            **predictions  # Unpack predictions dictionary to fill in the row
        }
        prediction_table = pd.concat([prediction_table, pd.DataFrame([new_row])], ignore_index=True)

    return prediction_table

# Example usage:
# Load your data
#data = pd.read_csv('data_new.csv')
data = pd.read_csv('data_with_covarites.csv')
data['data_iniSE'] = pd.to_datetime(data['data_iniSE'])  # Ensure datetime is the correct dtype

# Assuming 'casos' and other relevant features are correctly formatted and present
window_size = 313
results_df = predict_rf(data, window_size)
print(results_df.head())


results_df.to_csv('rf.csv', index=False)

from xgboost import XGBRegressor
import pandas as pd

# Load the new data with additional covariates
data = pd.read_csv('data_with_covarites.csv')
data['data_iniSE'] = pd.to_datetime(data['data_iniSE'])  # Ensure datetime is the correct dtype

# Assuming 'data' is prepared with 'casos', 'umidmed', 'tempmed'
# Add required columns to the results DataFrame
columns = ['date', 'real_cases', '1_week_prediction', '2_week_prediction', '3_week_prediction',
           '4_week_prediction', '8_week_prediction', '12_week_prediction']
results_df = pd.DataFrame(columns=columns)

# Initialize the XGBoost model
xgb_model = XGBRegressor(n_estimators=5000)

# Define the window size
window_size = 313

# Ensure the loop has enough data to form a window and predict up to 12 weeks ahead
for start in range(len(data) - window_size):  # adjusted for max forecast horizon
    end = start + window_size

    # Training data for the current window
    X_train = data.iloc[start:end].drop(columns=['data_iniSE', 'casos'])
    y_train = data.iloc[start:end]['casos']

    # Fit the model
    xgb_model.fit(X_train, y_train)

    # Prepare a dictionary to hold predictions
    predictions = {}

    # Predict for 1, 2, 3, 4, 8, and 12 weeks ahead
    for weeks_ahead in [1, 2, 3, 4, 8, 12]:
        if end + weeks_ahead - 1 < len(data):  # Ensure index exists
            X_test = data.iloc[end + weeks_ahead - 1:end + weeks_ahead].drop(columns=['data_iniSE', 'casos'])
            prediction = xgb_model.predict(X_test)[0]
            predictions[f'{weeks_ahead}_week_prediction'] = prediction

    # Create new row for the results DataFrame
    new_row = {
        'date': data.iloc[end]['data_iniSE'],
        'real_cases': data.iloc[end]['casos'],
        **predictions  # Include predictions from the dictionary
    }
    results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)

# Print the first few rows of the results to check
print(results_df.head())
