import os
import pandas as pd
import numpy as np

"""
    Generates feature vectors for each team in the DataFrame, grouped by player ID ('puuid'). 
    Each vector consists of sequences listing each champion and their corresponding items 
    followed by a set of unique augments used by the team. 
    If the number of unique augments is less than three, 'nan' is used to fill the remaining slots.

    Arguments:
    df (pandas.DataFrame): The complete DataFrame containing data for a match.

    Returns:
    list: A list of feature vectors, one for each team. 
    Each feature vector is a list containing the names of champions and
    their items, ending with three slots for augments, filled as available.
"""


def create_team_feature_vectors(df):
    # Remove unnecessary columns to focus on relevant data
    df_cleaned = df.drop(['Unnamed: 0', 'bigger_region', 'region', 'tier', 'placement',
                          'champion_rarity', 'champion_tier'], axis=1)

    # Group the dataframe by player ID ('puuid')
    grouped = df_cleaned.groupby('puuid')

    # Initialize a list to store the feature vectors for each team
    all_teams_feature_vectors = []

    # Iterate over each group
    for puuid, group in grouped:
        # Initialize a list for the team's feature vector
        team_feature_vector = []

        # Append champion and their items to the feature vector
        for index, row in group.iterrows():
            team_feature_vector.extend([row['champion_name'],
                                        row['item_1'] if pd.notnull(row['item_1']) else np.nan,
                                        row['item_2'] if pd.notnull(row['item_2']) else np.nan,
                                        row['item_3'] if pd.notnull(row['item_3']) else np.nan])

        # Collect unique augments at the end of the vector using a set to eliminate duplicates
        unique_augments = set(group['augment_1'].dropna().tolist() +
                              group['augment_2'].dropna().tolist() +
                              group['augment_3'].dropna().tolist())
        # Ensure there are exactly three augment slots
        unique_augments = list(unique_augments) + [np.nan] * (3 - len(unique_augments))

        # Append unique augments to the feature vector
        team_feature_vector.extend(unique_augments[:3])

        # Add this team's feature vector to the main list
        all_teams_feature_vectors.append(team_feature_vector)

    # Return the list of all teams' feature vectors
    return all_teams_feature_vectors


if __name__ == '__main__':
    input_folder = 'C:/Users/foktp/Desktop/pr_kom/matches'
    csv_files = [file for file in os.listdir(input_folder) if file.endswith('.csv')]
    all_pairs = []
    for file in csv_files:
        file_path = os.path.join(input_folder, file)
        df = pd.read_csv(file_path)
        pairs = create_team_feature_vectors(df)
        all_pairs.extend(pairs)

    print(all_pairs[:10])
