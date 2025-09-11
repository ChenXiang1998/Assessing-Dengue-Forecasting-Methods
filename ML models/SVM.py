import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('data_new.csv')
data['data_iniSE'] = pd.to_datetime(data['data_iniSE'])
data = data[['data_iniSE', 'casos']]  # Assuming 'casos' is the column for cases

# Prepare a DataFrame to store the results
results_df = pd.DataFrame()

# Initialize the scaler and SVR model
scaler = StandardScaler()
svm_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)

# Define the window size
window_size = 313

# Start the moving-window prediction
# Ensure that there is enough data to predict up to 12 weeks ahead
for start in range(len(data) - window_size):
    end = start + window_size

    # Training data for the current window
    X_train = data['casos'][start:end].values.reshape(-1, 1)
    y_train = data['casos'][start+1:end+1].values  # Shift target by 1 to predict the next step

    # Scale the features
    X_train_scaled = scaler.fit_transform(X_train)

    # Fit the model
    svm_model.fit(X_train_scaled, y_train)

    # Prepare to collect predictions for 1, 2, 3, 4, 8, and 12 weeks ahead
    predictions = {}
    for weeks_ahead in [1, 2, 3, 4, 8, 12]:
        if end + weeks_ahead - 1 < len(data):  # Ensure we have data to predict
            X_test = data['casos'][end + weeks_ahead-1].reshape(1, -1)
            X_test_scaled = scaler.transform(X_test)
            prediction = svm_model.predict(X_test_scaled)[0]
            predictions[f'Predicted_Cases_Week{weeks_ahead}'] = prediction

    # Append to the results DataFrame
    new_row = pd.DataFrame({
        'date': [data['data_iniSE'].iloc[end]],
        'Real_Cases': [data['casos'].iloc[end]],
        **{k: [v] for k, v in predictions.items()}
    })
    results_df = pd.concat([results_df, new_row], ignore_index=True)

# Display the results
print(results_df.head())
