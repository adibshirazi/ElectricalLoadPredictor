import os
import pandas as pd
from persiantools.jdatetime import JalaliDateTime

# Function to convert Persian date to Gregorian date
def convert_to_gregorian(persian_date):
    try:
        if pd.isna(persian_date):
            return None
        j_date = JalaliDateTime.strptime(str(persian_date), '%Y/%m/%d')
        return j_date.to_gregorian().strftime('%Y-%m-%d')
    except Exception as e:
        print(f"Error converting date {persian_date}: {e}")
        return None

# Directory containing the Excel files
directory = 'data'  # Adjust this path as per your directory structure

# Initialize an empty list to store DataFrames
data_frames = []

# Loop through all the files in the directory
try:
    for filename in os.listdir(directory):
        if filename.endswith(".xlsx"):
            # Load the Excel file
            file_path = os.path.join(directory, filename)
            df = pd.read_excel(file_path)
            # Check if 'date' column exists
            if 'date' in df.columns:
                # Convert Persian date to Gregorian date
                df['Gregorian Date'] = df['date'].apply(convert_to_gregorian)
                # Save the updated DataFrame back to the Excel file
                df.to_excel(file_path, index=False)
                # Append the DataFrame to the list
                data_frames.append(df)
            else:
                print(f"Warning: File {filename} does not contain 'date' column.")

    # Concatenate all the DataFrames
    if data_frames:
        data = pd.concat(data_frames, ignore_index=True)
        # Display the first few rows of the combined data
        print(data.head())
    else:
        print("No data frames were concatenated.")

except FileNotFoundError:
    print(f"Directory '{directory}' not found.")
except Exception as e:
    print(f"An error occurred: {e}")
