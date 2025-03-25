import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np

# Directory containing the CSV files (update this path)
directory = "results/gemma_gsm8k/"

# Collect all CSV files, including the main file
files = [f for f in os.listdir(directory) if f.endswith('.csv')]
files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]) if '_' in x else 0)  # Sort numerically

# Initialize DataFrame to store results
matrix = pd.DataFrame()

for i, file in enumerate(files):
    filepath = os.path.join(directory, file)
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Add 0 to stadardize shape of the column
    distances = []
    for _ in range(i):
        distances.append(np.nan)

    # Extract average_distance values
    for line in lines[3:]:  # Skip header lines
        if line.startswith("Layer"):
            break  # Stop at the pruning recommendation line
        parts = line.strip().split(',')
        distances.append(float(parts[2]))
    
    # Add to DataFrame (column name = file name)
    column_name = file.replace('.csv', '')
    matrix[column_name] = distances

# Set row labels as blocks (e.g., "1-2", "2-3")
blocks = [f"{i}-{i+1}" for i in range(1, 26)]
matrix.index = [i+1 for i in range(25)]

print(matrix)

# Define the colormap and set NaN values to white
viridis = mpl.colormaps['viridis'].copy()
viridis.set_bad(color="white")

# Convert DataFrame to NumPy array
matrix_array = matrix.values  # Shape: (25 layer blocks) x (N files)

# Generate shortened labels for files
column_labels = []
for col_name in matrix.columns:
    if col_name == "layer_distances_gemma_1b":
        column_labels.append("base")
    else:
        label = col_name.split("_")[-1]
        column_labels.append(label)

# Plot heatmap
plt.figure(figsize=(14, 8))
sns.heatmap(
    matrix_array,
    cmap="viridis",
    annot=False,
    cbar_kws={"label": "Average Distance"},
    yticklabels=matrix.index,
    xticklabels=column_labels,
)

plt.title("Average Layer Distances Across Configurations", fontsize=16)
plt.xlabel("Number of Layers to Skip", fontsize=16)
plt.ylabel("Layer Number, l", fontsize=16)
plt.xticks(rotation=45, ha="right")

plt.savefig(directory+"visualize")

plt.tight_layout()
plt.show()