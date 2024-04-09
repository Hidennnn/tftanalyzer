import pandas as pd
import os
import numpy as np

"""
    Reads all CSV files from a specified folder, combines them into a single DataFrame, 
    and saves the result to a new CSV file.

    This function is useful for aggregating data from multiple CSV files into one file, 
    making it easier to handle large datasets or prepare data for analysis.

    Arguments:
    input_folder_path (str): Path to the folder containing CSV files.
    output_file_path (str): Path where the combined CSV file will be saved.
"""
def combine_csv_files(input_folder_path, output_file_path):
    # Retrieve a list of all CSV file names in the specified directory
    csv_files = [file for file in os.listdir(input_folder_path) if file.endswith('.csv')]
    df_list = []  # List to store DataFrames

    # Iterate over each CSV file, read it into a DataFrame, and append to the list
    for file in csv_files:
        file_path = os.path.join(input_folder_path, file)  # Construct full path to file
        df = pd.read_csv(file_path)  # Read CSV into DataFrame
        df_list.append(df)  # Append DataFrame to list

    # Concatenate all DataFrames from the list into a single DataFrame
    combined_df = pd.concat(df_list, ignore_index=True)

    # Save the combined DataFrame to a CSV file at the specified output path
    combined_df.to_csv(output_file_path, index=False)


"""
    Creates target-context pairs from a DataFrame for use in Word2Vec training.

    This function iterates over each row of the DataFrame and pairs each 'champion_name' 
    with items and augments associated with it. These pairs are used to train Word2Vec models 
    to understand the context or association between champions and their items/augments.

    Arguments:
    df (pandas.DataFrame): DataFrame containing columns for 
    'champion_name', 'item_1', 'item_2', 'item_3', 'augment_1', 'augment_2', and 'augment_3'.

    Returns:
    list: A list of tuples, where each tuple contains a target (champion_name) 
    and a context (an item or augment associated with the champion).
"""
def create_target_context_pairs(df):
    # Remove unnecessary columns to focus on relevant data
    df = df.drop(
        ['Unnamed: 0', 'puuid', 'bigger_region', 'region', 'tier', 'placement',
         'champion_rarity', 'champion_tier'], axis=1)

    # Initialize a list to store the target-context pairs
    target_context_pairs = []

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # The target is the name of the champion
        target = row['champion_name']

        # Loop through the items and augments columns to collect contexts
        for context in [row['item_1'], row['item_2'], row['item_3'], row['augment_1'], row['augment_2'],
                        row['augment_3']]:
            # Only add the context if it is not null
            if pd.notnull(context):
                # Append the (target, context) pair to the list
                target_context_pairs.append((target, context))

    # Return the list of pairs
    return target_context_pairs

if __name__ == '__main__':
    input_folder = 'C:/Users/foktp/Desktop/pr_kom/matches'
    output_file = 'C:/Users/foktp/Desktop/pr_kom/combined.csv'
    combine_csv_files(input_folder, output_file)

    df = pd.read_csv(output_file)
    pairs = []
    pairs = create_target_context_pairs(df)
    print(pairs[:10])
