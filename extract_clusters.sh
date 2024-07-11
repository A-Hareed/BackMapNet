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
    cluster_id=$(echo $cluster_info | cut -d':' -f1 | cut -d' ' -f2)
    frames=$(echo $cluster_info | cut -d':' -f2 | tr -d '[],')
    mkdir -p cluster_${cluster_id}
    for frame in $frames; do
        echo 1 | gmx trjconv -s md_0_1.tpr -f fitted_protein.xtc -o cluster_${cluster_id}/frame_${frame}.pdb -dump $frame
        done
done

