import numpy as np

import numpy as np

def create_nonoverlapping_windows_endaligned(data, window_size):
    """
    Create non-overlapping windows, except the last one:
    - If fewer than window_size columns remain at the end,
      take those remaining columns and fill the start of the window
      with the preceding columns (slight overlap).
    """
    data = np.asarray(data)
    n_samples, total_len = data.shape

    num_full_windows = total_len // window_size
    remainder = total_len % window_size

    feat_arr = []

    # Full windows first
    for i in range(num_full_windows):
        start = i * window_size
        end = start + window_size
        feat_arr.append(data[:, start:end])

    # Handle the last partial window
    if remainder > 0:
        overlap_needed = window_size - remainder
        start_overlap = total_len - window_size
        last_window = data[:, start_overlap:total_len]
        feat_arr.append(last_window)

    feat_arr = np.array(feat_arr)
    return feat_arr
