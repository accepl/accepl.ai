import os
import pandas as pd

# Check if the file exists before trying to load it
file_path = 'data.csv'
if os.path.exists(file_path):
    data = pd.read_csv(file_path)
else:
    print(f"Warning: {file_path} not found. Using a dummy dataset for testing.")
    # Create a dummy dataset if the file does not exist (for testing)
    data = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6],
        'label': [0, 1, 0]
    })

# Rest of your training logic
print(data.head())  # Just to check if the data is loaded correctly
# Proceed with model training...
