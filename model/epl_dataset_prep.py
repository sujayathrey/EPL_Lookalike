import os
import pandas as pd

# Path to the preprocessed folder
preprocessed_path = "EPL_Players_Preprocessed_224x224_60"

# Create and populate a dataframe containing the image path and the label
df = pd.DataFrame(columns=['img_path', 'label'])

# Iterate through each player's folder
for player_folder in os.listdir(preprocessed_path):
    player_folder_path = os.path.join(preprocessed_path, player_folder)
    if os.path.isdir(player_folder_path):  # Ensure it's a directory
        for img_file in os.listdir(player_folder_path):
            img_path = os.path.join(player_folder_path, img_file)
            # Add the image path and label (folder name) to the dataframe
            df.loc[df.shape[0]] = [img_path, player_folder]

# Shuffle the dataset
df = df.sample(frac=1).reset_index(drop=True)

# Print a preview of the created dataframe
#print(df.head())

# Save the dataframe to a CSV file for future use (optional)
df.to_csv("epl_player_data_224x224_60.csv", index=False)

print("Dataframe created and saved successfully!")
print(df.shape)
