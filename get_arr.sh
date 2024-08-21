#!/bin/bash



# Extract frames for each cluster using the Python script
clusters=$(python3 extract_cluster_frames.py)

# Check if clusters variable is empty
if [[ -z "$clusters" ]]; then
    echo "No clusters found. Please check the cluster.log file and the Python script."
    exit 1
fi

# Convert the extracted frames into an array
IFS=$'\n' read -rd '' -a clusters_array <<< "$clusters"

# Loop over each cluster and its frames
for cluster_info in "${clusters_array[@]}"; do
    # Extract cluster ID
    cluster_id=$(echo $cluster_info | cut -d':' -f1 | cut -d' ' -f2)
    # Define the directory for the cluster
    cluster_dir="cluster_backbone_${cluster_id}"
    
    # Check if the directory exists
    if [ -d "$cluster_dir" ]; then
        echo "Directory $cluster_dir does exist"
        for pdb_file in "$cluster_dir"/*.pdb; do
            echo "name of pdb file  $pdb_file "
            echo "cluster_${cluster_id}.npy"
            python3 pdb2arr.py $pdb_file  cluster_${cluster_id}.npy
        done
    else
        echo "Directory $cluster_dir does not exist"
    fi
done
