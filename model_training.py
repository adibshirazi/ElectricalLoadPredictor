import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input, LSTM, GRU
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import jdatetime

# Check for GPU and set memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Load the Excel files
file1 = 'sorted_station1_data.xlsx'
file2 = 'sorted_station2_data.xlsx'

# Read the Excel files into DataFrames
print("Loading Excel files...")
df_station1 = pd.read_excel(file1)
df_station2 = pd.read_excel(file2)

# Combine the two dataframes on the date column
print("Combining DataFrames...")
df_combined = pd.merge(df_station1, df_station2, on='date', suffixes=('_Station1', '_Station2'))

# Verify the column names
print("Column names in df_combined:")
print(df_combined.columns)

# Prepare the dataset for time series prediction
def prepare_data(df, n_steps, n_lags=3):
    X, y = [], []
    for i in range(n_steps, len(df) - n_lags):
        seq_x = df[i-n_steps:i]
        seq_y = df[i:i+n_lags]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y).reshape(-1, n_lags * df.shape[1])

# Define the number of steps and lags
n_steps = 24
n_lags = 1

# Add Lag Features
def add_lag_features(df, n_lags=3):
    for lag in range(1, n_lags+1):
        for i in range(1, 25):
            df[f'Load_Station1_H{i}_lag_{lag}'] = df[f'Load_Station1_H{i}'].shift(lag)
            df[f'Load_Station2_H{i}_lag_{lag}'] = df[f'Load_Station2_H{i}'].shift(lag)
    return df.fillna(method='bfill')

df_combined = add_lag_features(df_combined, n_lags)
data = df_combined.drop(columns='date').values
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)
X, y = prepare_data(data_scaled, n_steps, n_lags)

# Define k-fold cross-validation
kf = KFold(n_splits=5)
mse_scores = []
r2_scores = []

# Learning rate schedule
initial_learning_rate = 0.001
lr_schedule = ExponentialDecay(
    initial_learning_rate,
    decay_steps=10000,
    decay_rate=0.9,
    staircase=True)

# Function to define the CNN model
def build_cnn_model():
    model = Sequential()
    model.add(Input(shape=(n_steps, X.shape[2])))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(y.shape[1], activation='relu'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss='mse')
    return model

# Function to define the LSTM model
def build_lstm_model():
    model = Sequential()
    model.add(Input(shape=(n_steps, X.shape[2])))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(32))
    model.add(Dropout(0.3))
    model.add(Dense(y.shape[1], activation='relu'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss='mse')
    return model

# Function to define the GRU model
def build_gru_model():
    model = Sequential()
    model.add(Input(shape=(n_steps, X.shape[2])))
    model.add(GRU(64, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(GRU(32))
    model.add(Dropout(0.3))
    model.add(Dense(y.shape[1], activation='relu'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss='mse')
    return model

# Train and evaluate using K-fold cross-validation
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    models = [build_cnn_model(), build_lstm_model(), build_gru_model()]
    ensemble_predictions = []

    for model in models:
        early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss', mode='min')

        print("Training model...")
        model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, verbose=1,
                  callbacks=[early_stopping, model_checkpoint])

        # Make predictions
        print("Making predictions...")
        y_pred = model.predict(X_test, verbose=1)
        ensemble_predictions.append(y_pred)

    # Average the predictions of the models
    y_pred_ensemble = np.mean(ensemble_predictions, axis=0)

    # Calculate the Mean Squared Error (MSE)
    print("Calculating MSE...")
    mse = mean_squared_error(y_test, y_pred_ensemble)
    mse_scores.append(mse)

    # Calculate R² score
    r2 = r2_score(y_test, y_pred_ensemble)
    r2_scores.append(r2)

# Calculate overall average scores
average_mse = np.mean(mse_scores)
average_r2 = np.mean(r2_scores)
accuracy_percentage = average_r2 * 100

print(f"Average Mean Squared Error: {average_mse}")
print(f"Average R² Score: {average_r2}")
print(f"Average Accuracy Percentage: {accuracy_percentage}%")

# Inverse transform the predictions
print("Inverse transforming predictions...")

# Generate future dates for the year 1403
print("Generating future dates for the year 1403...")
future_dates = []
for month in range(1, 13):  # 12 months
    for day in range(1, 32):  # Up to 31 days
        try:
            date = jdatetime.datetime(1403, month, day)
            future_dates.append(date.strftime('%Y/%m/%d'))
        except ValueError:
            continue  # Skip invalid dates

# Create a DataFrame for the predictions
print("Creating predictions DataFrame...")
hourly_columns = [f'Load_Station1_H{i+1}' for i in range(24)] + [f'Load_Station2_H{i+1}' for i in range(24)]
predictions_df = pd.DataFrame(np.zeros((len(future_dates), len(hourly_columns))), columns=hourly_columns)
predictions_df.insert(0, 'date', future_dates[:len(predictions_df)])  # Match the length of predictions

# Ensure the length of y_pred_ensemble matches the length of future_dates
if len(y_pred_ensemble) > len(future_dates):
    y_pred_ensemble = y_pred_ensemble[:len(future_dates)]
elif len(y_pred_ensemble) < len(future_dates):
    future_dates = future_dates[:len(y_pred_ensemble)]
    predictions_df = predictions_df.iloc[:len(y_pred_ensemble)]

# Update predictions_df with actual predictions
num_columns = len(hourly_columns)  # Number of columns excluding the date column
for i in range(len(future_dates)):
    inverse_transformed = scaler.inverse_transform(y_pred_ensemble[i].reshape(1, -1)).flatten()
    if len(inverse_transformed) != num_columns:
        raise ValueError(f"Mismatch in number of columns: expected {num_columns}, got {len(inverse_transformed)}")
    predictions_df.iloc[i, 1:] = inverse_transformed

# Save the predictions to an Excel file
print("Saving predictions to Excel file...")
predictions_df.to_excel('predictions_1403.xlsx', index=False)

print("Predictions saved successfully.")
