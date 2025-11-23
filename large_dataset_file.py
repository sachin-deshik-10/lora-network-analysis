import pandas as pd
import numpy as np

# Set a seed for reproducibility
np.random.seed(42)

# Define the number of samples and features
num_samples = 10000
num_features = 10

# Generate synthetic data
data = {
    f'feature_{i}': np.random.rand(num_samples) * 100 for i in range(num_features)
}

# Add a 'modulation' column with random categorical data
data['modulation'] = np.random.choice(['QAM', 'PSK', 'FSK'], num_samples)

# Add a 'signal_strength' column as the target, which is a continuous variable
data['signal_strength'] = np.random.rand(num_samples) * 100

# Create a DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('D:/Internship/large_dataset.csv', index=False)

print("large_dataset.csv file has been created successfully.")

