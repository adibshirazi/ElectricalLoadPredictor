import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from persiantools.jdatetime import JalaliDate

# Function to parse Persian date to Gregorian
def parse_date(date_str):
    try:
        if isinstance(date_str, str):
            year, month, day = map(int, date_str.split('/'))
            gregorian_date = JalaliDate(year, month, day).to_gregorian()
            return pd.Timestamp(gregorian_date)
        elif isinstance(date_str, float) and np.isnan(date_str):
            return pd.Timestamp('NaT')
        else:
            return pd.Timestamp('NaT')
    except Exception as e:
        print(f"Error in parsing date '{date_str}': {e}")
        return pd.Timestamp('NaT')

# Load the separated station data
station1_data_path = 'processed_station1_data.xlsx'
station2_data_path = 'processed_station2_data.xlsx'

station1_data = pd.read_excel(station1_data_path)
station2_data = pd.read_excel(station2_data_path)

# Reset index to turn 'date' into a regular column
station1_data.reset_index(inplace=True)
station2_data.reset_index(inplace=True)

# Concatenate the datasets assuming same columns
data = pd.concat([station1_data, station2_data])

# Check column names to verify 'date' column existence
print(data.columns)

# Ensure 'date' column exists before further operations
if 'date' in data.columns:
    try:
        # Apply parse_date function to 'date' column
        data['Date'] = data['date'].apply(parse_date)
        
        # Drop unnecessary columns if they exist
        columns_to_drop = ['index', 'date']  # Adjust columns to drop as per your dataset
        data.drop(columns=columns_to_drop, inplace=True)
        
        # Sort and set index by 'Date'
        data.set_index('Date', inplace=True)
        data.sort_index(inplace=True)
        
        # Fill missing values
        data.fillna(method='ffill', inplace=True)
        
        # Prepare features and target
        X = data.drop(columns=['Load_Station1_H1', 'Load_Station1_H10', 'Load_Station1_H11', 'Load_Station1_H12', 'Load_Station1_H13'])  # Adjust columns as per your dataset
        y = data['Load_Station1_H1']  # Adjust as per your target variable
        
        # Normalize the features
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        # Reshape the data for LSTM [samples, time steps, features]
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
        
        # Build the LSTM model
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        
        # Train the model
        history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test), verbose=2, shuffle=False)
        
        # Evaluate the model
        loss = model.evaluate(X_test, y_test, verbose=0)
        print(f'Model Loss: {loss}')
        
        # Predict future power load
        future_dates = pd.date_range(start=data.index.max(), periods=365*24, freq='H')
        future_data = pd.DataFrame(index=future_dates)
        future_data['Year'] = future_data.index.year
        future_data['Month'] = future_data.index.month
        future_data['Day'] = future_data.index.day
        future_data['Hour'] = future_data.index.hour
        
        # Assuming Load_Station1_H1 to Load_Station1_H24 are available for prediction
        future_X = future_data[['Year', 'Month', 'Day', 'Hour']]
        for i in range(1, 25):
            future_X[f'Load_Station1_H{i}'] = 0  # Placeholder, adjust as needed
        
        # Normalize the features
        future_X_scaled = scaler.transform(future_X)
        
        # Reshape the data for LSTM [samples, time steps, features]
        future_X_reshaped = future_X_scaled.reshape((future_X_scaled.shape[0], 1, future_X_scaled.shape[1]))
        
        # Predict using the model
        future_load = model.predict(future_X_reshaped)
        
        # Add predictions to future_data
        future_data['Predicted Load'] = future_load
        
        # Save the future predictions to an Excel file
        future_output_file_path = 'future_predictions.xlsx'
        future_data.to_excel(future_output_file_path, index_label='Date')
        
        print('Prediction completed and saved to future_predictions.xlsx.')
    
    except Exception as e:
        print(f"Error occurred: {e}")

else:
    print("'date' column not found in the DataFrame. Check your data loading and processing steps.")
