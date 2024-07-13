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
    mkdir -p cluster_backbone_${cluster_id}
    for frame in $frames; do

        output_file="cluster_backbone_${cluster_id}/frame_${frame}.pdb"
        if [ ! -f "$output_file" ]; then
            echo 4 | gmx trjconv -s md_0_1.tpr -f fitted_protein.xtc -o "$output_file" -dump $frame 2>&1 | tee -a gromacs.log
            if [ $? -ne 0 ]; then
                echo "Error processing frame $frame for cluster $cluster_id" >> gromacs.log
            fi
        else
            echo "File $output_file already exists. Skipping frame $frame for cluster $cluster_id." >> gromacs2.log 2>&1
        fi
    done
done
