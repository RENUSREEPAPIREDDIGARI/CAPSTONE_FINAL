import random

import pandas as pd

# Read the existing CSV file
df = pd.read_csv('data_mapping.csv')

# Add platform column with random values
platforms = ['Instagram', 'Twitter', 'LinkedIn']
df['platform'] = [random.choice(platforms) for _ in range(len(df))]

# Save the updated DataFrame back to CSV
df.to_csv('data_mapping.csv', index=False)
print("Data mapping file updated successfully with platform information!") 