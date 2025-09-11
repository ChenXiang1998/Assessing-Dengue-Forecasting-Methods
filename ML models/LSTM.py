import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

def prepare_sequences(data, window_size):
    """ Prepare sequences for LSTM training """
    X, y = [], []
    for i in range(len(data) - window_size):
        seq_x = data[i:(i + window_size)]
        seq_y = data[i + window_size]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y).reshape(-1, 1)

def predict_lstm(data, window_size):
    # Scale 'casos' data for LSTM using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['casos'].values.reshape(-1,1))

    # Prepare data for LSTM
    X, y = prepare_sequences(scaled_data, window_size)

    # Define LSTM model
    model = Sequential([
        LSTM(1000, activation='relu', input_shape=(window_size, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Fit the LSTM model
    model.fit(X, y, epochs=50, verbose=0)

    # Initialize the results DataFrame
    prediction_table = pd.DataFrame()

    # Start the moving-window prediction
    for i in range(len(data) - window_size):  # ensures space for predictions
        # Prepare input sequence for prediction
        input_seq = scaled_data[i:i + window_size]
        input_seq = input_seq.reshape((1, window_size, 1))

        # Predict the next 12 weeks
        predictions = []
        for weeks_ahead in [1, 2, 3, 4, 8, 12]:
            future_index = i + window_size + weeks_ahead - 1
            if future_index < len(data):
                prediction = model.predict(input_seq, verbose=0)[0][0]
                prediction = scaler.inverse_transform([[prediction]])[0][0]
                predictions.append(prediction)

        new_row = {
            'Datetime': data.iloc[i + window_size]['datetime'],
            'Real_Cases': data.iloc[i + window_size]['casos'],
            **{f'Predicted_Cases_Week{w}': p for w, p in zip([1, 2, 3, 4, 8, 12], predictions)}
        }
        prediction_table = pd.concat([prediction_table, pd.DataFrame([new_row])], ignore_index=True)

    return prediction_table

data = pd.read_csv('data_new.csv')
data['datetime'] = pd.to_datetime(data['data_iniSE'])
window_size = 313  # Fixed window size
results_df = predict_lstm(data, window_size)
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

results_df.to_csv('LSTM_1000.csv', index=False)  # index=False means do not write row names (index)
