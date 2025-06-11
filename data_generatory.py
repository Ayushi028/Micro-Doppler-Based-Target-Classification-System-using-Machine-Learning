import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split

def generate_data(n_samples=10000, test_size=0.6, random_state=42):
    np.random.seed(random_state)

    # Generate features
    frequency = np.random.uniform(1, 10, n_samples)
    amplitude = np.random.uniform(0, 1, n_samples)
    duration = np.random.uniform(0.1, 2, n_samples)

    # Generate labels (0 for bird, 1 for drone)
    labels = np.random.choice([0, 1], n_samples)

    # Adjust features based on labels
    frequency += labels * np.random.uniform(0, 5, n_samples)  # Drones tend to have higher frequency
    amplitude += labels * np.random.uniform(0, 0.5, n_samples)  # Drones tend to have higher amplitude
    duration -= labels * np.random.uniform(0, 0.5, n_samples)  # Birds tend to have longer duration

    # Create DataFrame
    df = pd.DataFrame({
        'frequency': frequency,
        'amplitude': amplitude,
        'duration': duration,
        'label': labels
    })

    df['class'] = df['label'].map({0: 'Bird', 1: 'Drone'})
    df['timestamp'] = [datetime.now() - timedelta(minutes=i) for i in range(n_samples)]

    # Split into train and test sets
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['label'])

    # Save to CSV files
    # train_df.to_csv('train_data.csv', index=False)
    test_df.to_csv('test_data1.csv', index=False)

    return train_df, test_df

# Example usage
train_data, test_data = generate_data()
print("Train Data:", train_data.head())
print("Test Data:", test_data.head())