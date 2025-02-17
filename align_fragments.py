import numpy as np

def normalize_fragments_per_axis(fragments):
    """
    Normalize each fragment along each axis using custom min-max normalization,
    and save custom_min and custom_range for reverse normalization.
    
    Parameters:
        fragments (ndarray): Array with shape (samples, number_of_atoms, 3).
    
    Returns:
        normalized_fragments (ndarray): Normalized fragments.
        custom_min (ndarray): Custom minimum values per axis (shape: (samples, 1, 3)).
        custom_range (ndarray): Custom range per axis (shape: (samples, 1, 3)).
    """
    # Compute per-axis min and max for each sample
    absolute_min = np.min(fragments, axis=1, keepdims=True)  # shape: (samples, 1, 3)
    absolute_max = np.max(fragments, axis=1, keepdims=True)  # shape: (samples, 1, 3)
    
    # Adjust by ±4 for each axis
    custom_min = absolute_min - 4
    custom_max = absolute_max + 4
    
    # Compute the custom range
    custom_range = custom_max - custom_min
    
    # Normalize fragments using broadcasting
    normalized_fragments = (fragments - custom_min) / custom_range
    
    return normalized_fragments, custom_min, custom_range

def align_fragment(fragment):
    """
    Align a fragment (with shape (number_of_atoms, 3)) using PCA.
    
    The process is:
      1. Compute the centroid and center the fragment.
      2. Compute the principal axes using SVD.
      3. Rotate the fragment so that its principal axes align with the coordinate axes.
    
    Parameters:
        fragment (ndarray): A fragment with shape (number_of_atoms, 3).
    
    Returns:
        aligned (ndarray): The aligned fragment.
        R (ndarray): The rotation matrix used.
        centroid (ndarray): The centroid of the original fragment.
    """
    # 1. Compute and subtract the centroid
    centroid = np.mean(fragment, axis=0)
    centered = fragment - centroid

    # 2. Compute SVD of the centered data.
    #    Note: For PCA, the principal axes are given by the rows of Vt.
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    R = Vt.T  # The rotation matrix that aligns the fragment's axes with the coordinate axes

    # Optional: Ensure a right-handed coordinate system
    if np.linalg.det(R) < 0:
        R[:, -1] *= -1

    # 3. Rotate the centered fragment
    aligned = np.dot(centered, R)
    
    return aligned, R, centroid

def transform_all_atom_with_reference(all_atom_fragment, ref_centroid, ref_R):
    """
    Transform an all-atom fragment using the reference centroid and rotation matrix
    from the coarse-grained model.
    
    Parameters:
        all_atom_fragment (ndarray): All-atom fragment (n_atoms, 3).
        ref_centroid (ndarray): Centroid from the coarse-grained model.
        ref_R (ndarray): Rotation matrix from the coarse-grained model.
    
    Returns:
        transformed (ndarray): The all-atom fragment transformed into the coarse-grained reference frame.
    """
    # Center the all-atom fragment using the reference centroid
    centered_all_atom = all_atom_fragment - ref_centroid
    # Apply the reference rotation
    transformed = np.dot(centered_all_atom, ref_R)
    return transformed


def process_fragments(fragments):
    """
    Normalize and then align each fragment.
    
    Parameters:
        fragments (ndarray): Array with shape (samples, number_of_atoms, 3).
    
    Returns:
        aligned_fragments (ndarray): Array of aligned fragments (same shape as input).
    """
    # First, normalize the fragments per axis
    normalized_fragments, custom_min, custom_range = normalize_fragments_per_axis(fragments)
    
    samples = normalized_fragments.shape[0]
    aligned_fragments = np.empty_like(normalized_fragments)
    
    # Align each fragment individually
    for i in range(samples):
        aligned, R, centroid = align_fragment(normalized_fragments[i])
        aligned_fragments[i] = aligned
    
    return aligned_fragments

# Example usage:
# Suppose you have protein backbone fragments stored in a numpy array with shape
# (samples, number_of_atoms, 3)
data = np.array([
    # Sample 1 (3 atoms)
    [[1, 10, 77], [6, 3, 69], [4, 19, 30]],
    # Sample 2 (3 atoms)
    [[2, 15, 80], [5, 8, 75], [3, 12, 35]]
])

# Process the fragments: first normalize, then align
aligned_data = process_fragments(data)

aligned_coarse, ref_R, ref_centroid = align_fragment(coarse_fragment)
aligned_all_atom = transform_all_atom_with_reference(all_atom_fragment, ref_centroid, ref_R)
print("Aligned Fragments:")
print(aligned_data)





plt.plot(LIN_uncentered[:,0], LIN_uncentered[:,1], c='r', marker='o')
plt.plot(J4N_uncentered[:,0], J4N_uncentered[:,1], c='b', marker='o',alpha=0.5)
plt.plot(TUP[:32,0], TUP[:32,1], c='g', marker='o',alpha=0.5)
plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.legend(['1LIN','1J4N','1TUP'])
plt.show()




import numpy as np
import matplotlib.pyplot as plt

# Sample Data (Replace with your actual data)
LIN_uncentered = np.random.rand(50, 2)  # Example data
J4N_uncentered = np.random.rand(50, 2)
TUP = np.random.rand(50, 2)

# High-resolution figure
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)  # Adjust figure size & resolution

# Plot each dataset with refined styles
ax.plot(LIN_uncentered[:,0], LIN_uncentered[:,1], 'ro-', markersize=5, linewidth=1.5, label='1LIN')
ax.plot(J4N_uncentered[:,0], J4N_uncentered[:,1], 'bo-', markersize=5, alpha=0.7, linewidth=1.5, label='1J4N')
ax.plot(TUP[:32,0], TUP[:32,1], 'go-', markersize=5, alpha=0.7, linewidth=1.5, label='1TUP')

# Improve labels & ticks
ax.set_xlabel("X Axis", fontsize=14, fontweight='bold')
ax.set_ylabel("Y Axis", fontsize=14, fontweight='bold')

# Fine-tune legend
ax.legend(fontsize=12, loc='best', frameon=True)

# Improve grid & axis appearance
ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
ax.tick_params(axis='both', which='major', labelsize=12)

# Display the plot
plt.show()
