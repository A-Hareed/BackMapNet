def normalize_fragments_per_axis(fragments):
    """
    Normalize each fragment along each axis using custom min-max normalization,
    and save custom_min and range (custom_max - custom_min) for reverse normalization.

    For each fragment and for each coordinate axis (x, y, z), the normalization is:
        normalized_value = (value - (min - 4)) / ((max + 4) - (min - 4))

    Parameters:
    fragments (ndarray): Array of fragments with shape (n_fragments, n_points, n_dimensions).

    Returns:
    normalized_fragments (ndarray): Normalized fragments with the same shape as input.
    custom_min (ndarray): Custom minimum values per axis with shape (n_fragments, 1, n_dimensions).
    custom_range (ndarray): Range (custom_max - custom_min) per axis with shape (n_fragments, 1, n_dimensions).
    """
    # Calculate the minimum and maximum along the points axis for each fragment and each coordinate
    absolute_min = np.min(fragments, axis=1, keepdims=True)  # shape: (n_fragments, 1, n_dimensions)
    absolute_max = np.max(fragments, axis=1, keepdims=True)  # shape: (n_fragments, 1, n_dimensions)
    
    # Adjust the min and max by ±4 for each axis
    custom_min = absolute_min - 4
    custom_max = absolute_max + 4
    
    # Calculate the range (custom_max - custom_min) for each axis
    custom_range = custom_max - custom_min
    
    # Normalize using the custom min-max per axis
    normalized_fragments = (fragments - custom_min) / custom_range
    
    return normalized_fragments, custom_min, custom_range
