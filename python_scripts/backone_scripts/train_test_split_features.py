import glob
import random
import numpy as np
import sys

pdb = sys.argv[1]
# Set the seed for reproducibility
random.seed(42)  # Choose your desired seed number

boxsize = 12.16946 * 10
boxsize = 1.0
# Find all numpy files matching the pattern
file_list = glob.glob('cluster_*.npy')
target_file_list = glob.glob('Feature_array/cluster_*_CG.npy')
# Shuffle the file list
random.shuffle(file_list)


# Function to extract cluster number from filename
def extract_cluster_number_CG(filename):
    # Extract the number between 'cluster_' and '_CG.npy'
    return int(filename.split('cluster_')[1].split('_CG')[0])


# Function to extract cluster number from filename
def extract_cluster_number(filename):
    # Extract the number between 'cluster_' and '.npy'
    return int(filename.split('_')[1].split('.')[0])


def extract_number(filename,j=1):
    # Split the string at underscores and extract the second part
    parts = filename.split('_')
    number = parts[j]
    if j ==1:
        number = number.split('.')[0]
    return int(number)
correct_target_lst = []
for i in range(len(file_list)):
    num = extract_number(file_list[i],1)
    for j in range(len(target_file_list)):
        num2 = extract_number(target_file_list[j],2)
        if num == num2:

            print(num==num2)
            print(num,num2)
            print(file_list[i],target_file_list[j])
            correct_target_lst.append(target_file_list[j])
#print(file_list,correct_target_lst)
# Determine the split ratio
split_ratio = 0.80  # 80% training, 20% testing
split_index = int(len(file_list) * split_ratio)

# Split the file list into training and testing files
#train_files = file_list[:split_index]
#test_files = file_list[split_index:]


train_files = correct_target_lst[:split_index]
test_files = correct_target_lst[split_index:]
print('the train list')
print(train_files)
print('the test list')
print(test_files)
# Initialize lists to hold training and testing data
X_train, X_test = [], []

# Load data from training files
for file in train_files:
    data = np.load(file).astype(float)
 #   data = data/boxsize

#    cluster_number = extract_cluster_number_CG(file)
#    cluster_column = np.full((data.shape[0], 1), cluster_number)
#    data_with_cluster = np.hstack((data, cluster_column))




    print(data.shape)
    X_train.append(data)
    print(len(X_train))
# Load data from testing files
for file in test_files:
    data = np.load(file).astype(float)
#    data = data/boxsize

#    cluster_number = extract_cluster_number_CG(file)
#    cluster_column = np.full((data.shape[0], 1), cluster_number)
#    data_with_cluster = np.hstack((data, cluster_column))



    X_test.append(data)
    print(len(X_test))

# Convert lists to numpy arrays
X_train = np.concatenate(X_train,axis=0).astype(float)
print(X_train.shape)
X_test = np.concatenate(X_test,axis=0).astype(float)
print(X_test)
np.save(f'training_features_{pdb}_subsetDim',X_train)
np.save(f'testing_features_{pdb}_subsetDim',X_test)

