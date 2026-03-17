import os
import numpy as np
import sys


def sliding_window_reconstruct(sequence, stride=12):
    """
    Reconstruct 1D sequence from overlapping windows by averaging overlaps.
    sequence shape: [num_windows, window_size]
    """
    num_windows, window_size = sequence.shape
    original_length = (num_windows - 1) * stride + window_size

    reconstructed = np.zeros(original_length, dtype=np.float64)
    counts = np.zeros(original_length, dtype=np.float64)

    for i in range(num_windows):
        start = i * stride
        end = start + window_size
        reconstructed[start:end] += sequence[i]
        counts[start:end] += 1.0

    counts[counts == 0.0] = 1.0
    reconstructed /= counts
    return reconstructed


def calculate_rmsd(array1, array2):
    if array1.shape != array2.shape:
        raise ValueError(f"Shape mismatch for RMSD: {array1.shape} vs {array2.shape}")
    diff = array1 - array2
    return float(np.sqrt(np.mean(np.square(diff))))


def reverse_normalize_fragments_per_axis(normalized_fragments, custom_min, custom_range):
    """
    Reverse min-max normalization with per-window/per-axis min and range.
    """
    return (normalized_fragments * custom_range) + custom_min


def regroup_window_major(flat_rows, num_frames):
    """
    Convert [num_windows*num_frames, ...] (window-major) to [num_frames, num_windows, ...].
    """
    total_rows = flat_rows.shape[0]
    if total_rows % num_frames != 0:
        raise ValueError(
            f"Cannot regroup {total_rows} rows into {num_frames} frames."
        )
    num_windows = total_rows // num_frames
    reshaped = flat_rows.reshape(num_windows, num_frames, *flat_rows.shape[1:])
    return np.transpose(reshaped, [1, 0] + list(range(2, reshaped.ndim)))


def infer_num_frames(total_rows, expected_length, window_size=384, stride=12):
    """
    Infer frame count using expected reconstructed length when provided.
    Falls back to 1 for backwards compatibility.
    """
    if expected_length is None:
        return 1
    if expected_length < window_size:
        return 1
    if (expected_length - window_size) % stride != 0:
        return 1
    windows_per_frame = ((expected_length - window_size) // stride) + 1
    if windows_per_frame <= 0:
        return 1
    if total_rows % windows_per_frame != 0:
        return 1
    return total_rows // windows_per_frame


def parse_optional_int(args, idx):
    if len(args) <= idx:
        return None
    try:
        return int(args[idx])
    except (TypeError, ValueError):
        return None


def main():
    if len(sys.argv) < 6:
        raise SystemExit(
            "Usage: python new_reverse_scaling.py pred.npy actual.npy custom_min.npy custom_range.npy "
            "chain_num [pdb_name] [frame_id] [expected_length] [num_frames]\n"
            "Use '-' for actual.npy in CG-only mode."
        )

    pred_path = sys.argv[1]
    actual_path = sys.argv[2]
    min_path = sys.argv[3]
    range_path = sys.argv[4]
    chain_num = sys.argv[5]

    pdb_name = sys.argv[6] if len(sys.argv) > 6 else None
    frame_id = sys.argv[7] if len(sys.argv) > 7 else None
    expected_length = parse_optional_int(sys.argv, 8)
    explicit_num_frames = parse_optional_int(sys.argv, 9)

    verbose = os.environ.get("RAMA_VERBOSE", "0") == "1"

    pred_norm = np.load(pred_path)
    has_actual = (actual_path != "-") and os.path.exists(actual_path)
    actual_norm = np.load(actual_path) if has_actual else None
    custom_min = np.load(min_path)
    custom_range = np.load(range_path)

    pred_3d = pred_norm.reshape(-1, 128, 3)
    actual_3d = actual_norm.reshape(-1, 128, 3) if has_actual else None
    min_3d = custom_min.reshape(-1, 1, 3)
    range_3d = custom_range.reshape(-1, 1, 3)

    n_rows = pred_3d.shape[0]
    actual_rows_ok = True if not has_actual else (actual_3d.shape[0] == n_rows)
    if (not actual_rows_ok) or min_3d.shape[0] != n_rows or range_3d.shape[0] != n_rows:
        raise ValueError(
            "Input row counts do not match: "
            f"pred={pred_3d.shape[0]}, actual={'NA' if actual_3d is None else actual_3d.shape[0]}, "
            f"custom_min={min_3d.shape[0]}, custom_range={range_3d.shape[0]}"
        )

    if explicit_num_frames is not None:
        num_frames = explicit_num_frames
    else:
        num_frames = infer_num_frames(n_rows, expected_length, window_size=384, stride=12)

    if num_frames <= 0:
        raise ValueError(f"Invalid num_frames inferred/provided: {num_frames}")

    original_pred_rows = reverse_normalize_fragments_per_axis(pred_3d, min_3d, range_3d).reshape(-1, 384)
    original_actual_rows = (
        reverse_normalize_fragments_per_axis(actual_3d, min_3d, range_3d).reshape(-1, 384)
        if has_actual
        else None
    )

    original_fragments_pred = regroup_window_major(original_pred_rows, num_frames)  # [F, W, 384]
    original_fragments_actual = (
        regroup_window_major(original_actual_rows, num_frames) if has_actual else None
    )  # [F, W, 384]

    if has_actual and verbose:
        diff_rows = original_fragments_actual.reshape(-1, 3) - original_fragments_pred.reshape(-1, 3)
        print(
            "max/min/mean diff before reconstruction:",
            diff_rows.max(axis=0),
            diff_rows.min(axis=0),
            diff_rows.mean(axis=0),
            diff_rows.shape,
        )
        print("regrouped shapes:", original_fragments_pred.shape, original_fragments_actual.shape)
        print("num_frames:", num_frames)

    # Reconstruct per frame to full chain coordinates
    recon_pred = []
    recon_actual = [] if has_actual else None
    for frame in range(num_frames):
        pred_full = sliding_window_reconstruct(original_fragments_pred[frame], stride=12)
        act_full = (
            sliding_window_reconstruct(original_fragments_actual[frame], stride=12)
            if has_actual
            else None
        )

        if expected_length is not None and pred_full.shape[0] != expected_length:
            raise ValueError(
                f"Reconstructed length {pred_full.shape[0]} does not match expected_length {expected_length}"
            )

        recon_pred.append(pred_full)
        if has_actual:
            recon_actual.append(act_full)

    yhat_array = np.asarray(recon_pred)
    actual_array = np.asarray(recon_actual) if has_actual else None

    if has_actual and verbose:
        diff_recon = yhat_array.reshape(-1, 3) - actual_array.reshape(-1, 3)
        print(
            "max/min/mean diff after reconstruction:",
            diff_recon.max(axis=0),
            diff_recon.min(axis=0),
            diff_recon.mean(axis=0),
            diff_recon.shape,
        )
        print("final reconstructed shapes:", yhat_array.shape, actual_array.shape)

    arr_rmsd = None
    if has_actual:
        lst_rmsd = []
        for frame in range(num_frames):
            rmsd_actual = calculate_rmsd(actual_array[frame].reshape(-1, 3), yhat_array[frame].reshape(-1, 3))
            if verbose:
                print(f"frame {frame} rmsd: {rmsd_actual}")
            lst_rmsd.append(rmsd_actual)
        arr_rmsd = np.asarray(lst_rmsd, dtype=np.float64)
        if verbose:
            print("rmsd shape:", arr_rmsd.shape)

    # Keep old outputs for compatibility with existing downstream scripts.
    legacy_suffix = os.path.basename(range_path)
    np.save(f"pred_chain{legacy_suffix}", yhat_array)
    if has_actual:
        np.save(f"actual_chain{legacy_suffix}", actual_array)

    # Also save stable, explicit names.
    name_parts = []
    if pdb_name:
        name_parts.append(str(pdb_name))
    if frame_id:
        name_parts.append(f"frame{frame_id}")
    name_parts.append(f"chain{chain_num}")
    name_parts.append(f"frames{num_frames}")
    tag = "_".join(name_parts)

    np.save(f"pred_{tag}.npy", yhat_array)
    if has_actual:
        np.save(f"actual_{tag}.npy", actual_array)
        np.save(f"rmsd_{tag}.npy", arr_rmsd)


if __name__ == "__main__":
    main()
