import glob
import random
import numpy as np

# Set the seed for reproducibility
random.seed(42)  # Choose your desired seed number

# Find all numpy files matching the pattern
file_list = glob.glob('cluster_*.npy')

# Shuffle the file list
random.shuffle(file_list)

# Determine the split ratio
split_ratio = 0.8  # 80% training, 20% testing
split_index = int(len(file_list) * split_ratio)

# Split the file list into training and testing files
train_files = file_list[:split_index]
test_files = file_list[split_index:]

# Initialize lists to hold training and testing data
X_train, X_test = [], []

# Load data from training files
for file in train_files:
    data = np.load(file)
    X_train.append(data)

# Load data from testing files
for file in test_files:
    data = np.load(file)
    X_test.append(data)

# Convert lists to numpy arrays
X_train = np.array(X_train)
X_test = np.array(X_test)

# Now X_train and X_test contain the data from selected files for training and testing
