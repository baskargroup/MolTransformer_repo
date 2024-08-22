import pandas as pd
import os

# Define the base path to your CSV files
csv_base_path = '/Users/tcpba/2024Spring/ss_test_data/SS_test_'

# Create a list to hold data from all CSV files
data_list = []


# Read all CSV files from SS_test_0.csv to SS_test_20.csv and append their data to the list
for i in range(2):
    file_path = f"{csv_base_path}{i}.csv"
    df = pd.read_csv(file_path)
    data_list.append(df)



# Concatenate all dataframes into one
full_data = pd.concat(data_list, ignore_index=True)

# Shuffle the entire dataset
shuffled_data = full_data.sample(frac=1).reset_index(drop=True)

# Calculate the number of rows per file
rows_per_file = len(shuffled_data) // 20

# Define the output folder
output_folder = '/Users/tcpba/2024Spring/ss_test_data/shuffled_data/'

# Ensure the output directory exists
os.makedirs(output_folder, exist_ok=True)

# Split the shuffled data into 20 roughly equal parts and save as new CSV files
for i in range(2):
    start_row = i * rows_per_file
    end_row = (i + 1) * rows_per_file if i < 19 else len(shuffled_data)
    
    output_file = os.path.join(output_folder, f'shuffled_data_part_{i+1}.csv')
    shuffled_data.iloc[start_row:end_row].to_csv(output_file, index=False)

print("Shuffling and splitting complete.")
