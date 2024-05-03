import pandas as pd
from scipy.io import arff
import os

# Step 1: Read ARFF file
file_path = os.path.join('datasets', 'iris.arff')
data, meta = arff.loadarff(file_path)

# Step 2: Convert to DataFrame
df = pd.DataFrame(data)

# Step 3: Write to CSV
csv_file_path = os.path.join('datasets', 'iris.csv')
df.to_csv(csv_file_path, index=False)
