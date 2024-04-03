import pandas as pd
import os

"""folder_path = 'C:/Users/foktp/Desktop/pr_kom/matches'
csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
df_list = []

for file in csv_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)
    df_list.append(df)

combined_df = pd.concat(df_list, ignore_index=True)

combined_df.to_csv('C:/Users/foktp/Desktop/pr_kom/combined.csv', index=False)"""


"""combined_df = pd.read_csv('C:/Users/foktp/Desktop/pr_kom/combined.csv')
combined_df.head()

combined_df = combined_df.drop(['Unnamed: 0', 'puuid', 'bigger_region', 'region', 'tier', 'placement', 'champion_rarity', 'champion_tier'], axis=1)
combined_df.head()

target_context_pairs = []

for index, row in combined_df.iterrows():
    target = row['champion_name']
    
    for context in [row['item_1'], row['item_2'], row['item_3'], row['augment_1'], row['augment_2'], row['augment_3']]:
        if pd.notnull(context):
            target_context_pairs.append((target, context))

print(target_context_pairs[:10])"""


def create_target_context_pairs(df):
    
    df = df.drop(['Unnamed: 0', 'puuid', 'bigger_region', 'region', 'tier', 'placement', 'champion_rarity', 'champion_tier'], axis=1)
    
    target_context_pairs = []
    
    for index, row in df.iterrows():
        target = row['champion_name']
        
        for context in [row['item_1'], row['item_2'], row['item_3'], row['augment_1'], row['augment_2'], row['augment_3']]:
            if pd.notnull(context):
                target_context_pairs.append((target, context))
                
    return target_context_pairs

folder_path = 'C:/Users/foktp/Desktop/pr_kom/matches'
csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

all_pairs = []

for file in csv_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)
    pairs = create_target_context_pairs(df)
    all_pairs.extend(pairs)

print(target_context_pairs[:10])






