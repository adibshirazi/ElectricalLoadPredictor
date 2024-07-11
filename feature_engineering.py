import pandas as pd

# Load the combined data file
data_path = 'combined_data.xlsx'
data = pd.read_excel(data_path)

# Check the structure of the data
print(data.head())

# Prepare lists to store processed data for each station
station1_data = []
station2_data = []

# Iterate over the rows in the dataframe
i = 0
while i < len(data):
    # Extract data for station1
    station1_row = {'date': data.loc[i, 'date']}
    for hour in range(1, 25):
        hour_column = f'H{hour}'
        if hour_column in data.columns:
            station1_row[f'Load_Station1_H{hour}'] = data.loc[i, hour_column]
    station1_data.append(station1_row)

    # Move to next row (skip the third row)
    i += 1

    if i < len(data):  # Check if we have more rows for station2
        # Extract data for station2
        station2_row = {'date': data.loc[i, 'date']}
        for hour in range(1, 25):
            hour_column = f'H{hour}'
            if hour_column in data.columns:
                station2_row[f'Load_Station2_H{hour}'] = data.loc[i, hour_column]
        station2_data.append(station2_row)

    # Move to the next set of rows for the next station1
    i += 2

# Create dataframes for each station
station1_df = pd.DataFrame(station1_data)
station2_df = pd.DataFrame(station2_data)

# Save the processed data to separate Excel files
station1_df.to_excel('processed_station1_data.xlsx', index=False)
station2_df.to_excel('processed_station2_data.xlsx', index=False)

print("Data processing completed. Station data saved to 'processed_station1_data.xlsx' and 'processed_station2_data.xlsx'.")
