# Power Load Prediction Using Deep Learning

This project aims to predict the power load for two stations using historical load data. The model employs deep learning techniques, including Convolutional Neural Networks (CNN), Long Short-Term Memory (LSTM), and Gated Recurrent Units (GRU), to forecast future load values. The prediction is done for the year 1403 in the Persian calendar.

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Data Preparation](#data-preparation)
4. [Model Architecture](#model-architecture)
5. [Training and Evaluation](#training-and-evaluation)
6. [Generating Predictions](#generating-predictions)
7. [Results](#results)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/adibshirazi/ElectricalLoadPredictor.git
    ```
2. Navigate to the project directory:
    ```bash
    cd ElectricalLoadPredictor
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Ensure the Excel files `sorted_station1_data.xlsx` and `sorted_station2_data.xlsx` are in the project directory.
2. Run the script:
    ```bash
    python model_training.py
    ```

## Data Preparation

1. **Loading Data**: The historical load data is loaded from two Excel files, `sorted_station1_data.xlsx` and `sorted_station2_data.xlsx`, into Pandas DataFrames.
2. **Combining Data**: The data from the two stations are combined into a single DataFrame based on the `date` column.
3. **Adding Lag Features**: Lag features are added to the DataFrame to include previous load values, which help the model learn temporal patterns.
4. **Scaling Data**: The combined data is scaled using MinMaxScaler to normalize the values for better model performance.

## Model Architecture

Three different deep learning models are used for prediction:

1. **Convolutional Neural Network (CNN)**: This model uses Conv1D layers to capture spatial patterns in the time series data, followed by MaxPooling, Flatten, Dense, and Dropout layers.
2. **Long Short-Term Memory (LSTM)**: The LSTM model uses LSTM layers to capture long-term dependencies in the time series data, followed by Dense and Dropout layers.
3. **Gated Recurrent Unit (GRU)**: The GRU model is similar to the LSTM model but uses GRU layers, which are computationally more efficient.

## Training and Evaluation

1. **K-Fold Cross-Validation**: The data is split into 5 folds, and each model is trained and evaluated on each fold to ensure robustness.
2. **Early Stopping and Model Checkpointing**: Early stopping is used to prevent overfitting, and the best model is saved using model checkpointing.
3. **Evaluation Metrics**: Mean Squared Error (MSE) and R² Score are used to evaluate model performance.

## Generating Predictions

1. **Future Date Generation**: Future dates for the year 1403 are generated in the Persian calendar.
2. **Inverse Transformation**: The predicted values are inverse transformed to their original scale.
3. **Saving Predictions**: The predictions are saved to an Excel file named `predictions_1403.xlsx`.

## Results

The model's performance is evaluated based on the average MSE and R² score across all folds. The final predictions for the year 1403 are saved in an Excel file, providing the forecasted power load for each hour and station.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
