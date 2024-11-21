import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file into a DataFrame

script_dir = os.path.dirname(os.path.abspath(__file__))
scraping_results_path = os.path.join(script_dir, '../data/cleaned_dataset.csv')
df = pd.read_csv(scraping_results_path)

# New Column for Price/m2 

df['Price/m2'] = df['Price'] / df['Livable Space (m2)']

df['Price/m2 Ratio'] = df['Price/m2'] / df['Price/m2'].max()

# Dropping outliers in Price/m2

df = df[(df['Price/m2'] >= (df['Price/m2'].mean() - 0.1 * df['Price/m2'].std())) & (df['Price/m2'] <= (df['Price/m2'].mean() + 2 * df['Price/m2'].std()))]

# Calculate the means and medians of 'Price Ratio' grouped by Column Type

df['Price Ratio'] = df['Price'] / df['Price'].max()

zip_code = df.groupby('Zip Code')['Price Ratio'].mean()
prop_type = df.groupby('Type of Property')['Price Ratio'].mean()
rooms_number = df.groupby('Number of Rooms')['Price Ratio'].mean()
livable_space = df.groupby('Livable Space (m2)')['Price Ratio'].mean()
land = df.groupby('Surface of the Land (m2)')['Price Ratio'].mean()
building = df.groupby('State of the Building')['Price Ratio'].mean()
consumption = df.groupby('Primary Energy Consumption (kWh/m2)')['Price Ratio'].mean()


# Map the means and medians values back to the DataFrame as a new columns

df['Zip Code Score'] = df['Zip Code'].map(zip_code)
df['Zip Code Score'] = df['Zip Code Score'] / df['Zip Code Score'].max()

df['Type of Property Score'] = df['Type of Property'].map(prop_type)
df['Type of Property Score'] = df['Type of Property Score'] / df['Type of Property Score'].max()

df['Number of Rooms Score'] = df['Number of Rooms'].map(rooms_number)
df['Number of Rooms Score'] = df['Number of Rooms Score'] / df['Number of Rooms Score'].max()

df['Livable Space (m2) Score'] = df['Livable Space (m2)'].map(livable_space)
df['Livable Space (m2) Score'] = df['Livable Space (m2) Score'] / df['Livable Space (m2) Score'].max()

df['Surface of the Land (m2) Score'] = df['Surface of the Land (m2)'].map(land)
df['Surface of the Land (m2) Score'] = df['Surface of the Land (m2) Score'] / df['Surface of the Land (m2) Score'].max()

df['State of the Building Score'] = df['State of the Building'].map(building)
df['State of the Building Score'] = df['State of the Building Score'] / df['State of the Building Score'].max()

df['Primary Energy Consumption (kWh/m2) Score'] = df['Primary Energy Consumption (kWh/m2)'].map(consumption)
df['Primary Energy Consumption (kWh/m2) Score'] = df['Primary Energy Consumption (kWh/m2) Score'] / df['Primary Energy Consumption (kWh/m2) Score'].max()

# Save to normalized_dataset

normalized_dataset_path = os.path.join(script_dir, '../data/normalized_dataset.csv')
df.to_csv(normalized_dataset_path, index=False)