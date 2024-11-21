import pandas as pd
import os 

# Read the CSV file into a DataFrame
script_dir = os.path.dirname(os.path.abspath(__file__))
scraping_results_path = os.path.join(script_dir, '../data/scraping_results.csv')
df = pd.read_csv(scraping_results_path)

#Cleaning

# Drop rows where either 'Price' or 'Number of Rooms' has missing values
df = df.dropna(subset=['Price', 'Number of Rooms'])


# Drop Type of Sale
df= df.drop(columns=['Type of Sale'])

# # Create the new 'Open Fire' column based on 'Any Fireplace ?'
# # Find the index of the current 'Any Fireplace ?' column

fireplaces_index = df.columns.get_loc('Any Fireplace ?')
df['Open Fire'] = df['Any Fireplace ?'].apply(lambda x: 0 if x == 0 else 1)
df.insert(fireplaces_index, 'Open Fire', df.pop('Open Fire'))
df = df.drop(columns=['Any Fireplace ?'])


# # Drop missing values in the 'Number of Facades' columns with 0
df = df.drop(columns=['Number of Facades'])

# Drop Terrace Area
df = df.drop(columns=['Terrace Area (m2)'])

# Drop Garden Area
df = df.drop(columns=['Garden Area (m2)'])

# Replace missing values in the 'Surface of the Land (m2)' columns with 0, dropping the rest
df.loc[df['Subtype of Property'] == 'apartment', 'Surface of the Land (m2)'] = df.loc[
    df['Subtype of Property'] == 'apartment', 'Surface of the Land (m2)'
].fillna(0)
df = df.dropna(subset=['Surface of the Land (m2)'])

# # Drop missing values in the 'Primary Energy Consumption (kWh/m2)' column
df['Primary Energy Consumption (kWh/m2)'] = df['Primary Energy Consumption (kWh/m2)'].fillna(0)

# Encode state of the building: Ordinal encoding, since categories have inherent order + drop null values
state_mapping = {
    'To restore': 0,
    'To renovate': 1,
    'To be done up': 2,
    'Good': 3,
    'Just renovated': 4,
    'As new': 5
}

df['State of the Building'] = df['State of the Building'].replace(state_mapping)
df['State of the Building'] = df['State of the Building'].fillna(0)

# Encode PEB: Ordinal encoding, since categories have inherent order + drop null values
PEB_mapping = {
    'G': 0,
    'F': 1,
    'E': 2,
    'D': 3,
    'C': 4,
    'B': 5,
    'A': 6
}

df['PEB'] = df['PEB'].replace(PEB_mapping)
df['PEB'] = df['PEB'].fillna(0)

# # Drop Construction Year column
df = df.drop(columns=['Construction Year'])

# # Drop the 'Url' column from the DataFrame (since not required and non-numerical values)
df = df.drop_duplicates(subset=['Price', 'Livable Space (m2)', 'Number of Rooms'])

# Drop outliers

df = df[(df['Price'] >= (df['Price'].mean() - 0.5 * df['Price'].std())) & (df['Price'] <= (df['Price'].mean() + 0.5 * df['Price'].std()))]
df = df[(df['Number of Rooms'] >= (df['Number of Rooms'].mean() - 7 * df['Number of Rooms'].std())) & (df['Number of Rooms'] <= (df['Number of Rooms'].mean() + 3 * df['Number of Rooms'].std()))]
df = df[(df['Primary Energy Consumption (kWh/m2)'] <= (df['Primary Energy Consumption (kWh/m2)'].mean() + 0.001 * df['Primary Energy Consumption (kWh/m2)'].std()))]
df = df[(df['Surface of the Land (m2)'] <= (df['Surface of the Land (m2)'].mean() + 5 * df['Surface of the Land (m2)'].std()))]
df = df[(df['Livable Space (m2)'] >= (df['Livable Space (m2)'].mean() - 1.2 * df['Livable Space (m2)'].std())) & (df['Livable Space (m2)'] <= (df['Livable Space (m2)'].mean() + 3 * df['Livable Space (m2)'].std()))]


# # # Set all values as int

df_droped = df.drop(columns=['Type of Property', 'Subtype of Property', 'Locality', 'Url'])
columns_list = df_droped.columns.tolist()

for column in columns_list:
    df[column] = df[column].astype(int)


# Saving

clean_dataset_path = os.path.join(script_dir, '../data/cleaned_dataset.csv')
df.to_csv(clean_dataset_path, index=False)

print(df.shape)