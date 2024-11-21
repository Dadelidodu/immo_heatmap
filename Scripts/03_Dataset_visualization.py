import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file into a DataFrame

script_dir = os.path.dirname(os.path.abspath(__file__))
normalized_dataset_path = os.path.join(script_dir, '../data/normalized_dataset.csv')
df = pd.read_csv(normalized_dataset_path)

# Select numerical columns for correlation
numerical_columns = ['Fully Equipped Kitchen','Furnished','Open Fire','Terrace','Garden','Swimming Pool','State of the Building Score','Primary Energy Consumption (kWh/m2) Score','Type of Property Score','Zip Code Score', 
    'Number of Rooms Score', 'Livable Space (m2) Score', 
    'Surface of the Land (m2) Score', 'Price'
]

# Compute the correlation matrix
correlation_matrix = df[numerical_columns].corr()

# Create a heatmap using seaborn
plt.figure(figsize=(10, 8))  # Set figure size
sns.heatmap(
    correlation_matrix, 
    annot=True,             # Annotate each cell with correlation value
    cmap='coolwarm',        # Use a diverging color palette
    fmt='.2f',              # Format the annotations to two decimal places
    linewidths=0.5          # Add space between cells
)

# Set titles and labels
plt.title("Correlation Map", fontsize=15)
plt.xticks(rotation=90)  # Rotate x-axis labels for readability
plt.yticks(rotation=0)   # Keep y-axis labels horizontal

# Display the heatmap
plt.tight_layout()
plt.show()