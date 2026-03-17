import argparse
import re
from pathlib import Path

import numpy as np

try:
    from mdplus.fast import rmsd_traj
except Exception:
    rmsd_traj = None


LEGACY_RE = {
    "pred": re.compile(r"^(?:ramaP_)?pred_chaincustom_range_B(\d+)_(.+)_chain(\d+)\.npy$"),
    "actual": re.compile(r"^(?:ramaP_)?actual_chaincustom_range_B(\d+)_(.+)_chain(\d+)\.npy$"),
}
MODERN_RE = {
    "pred": re.compile(r"^pred_(.+?)_frame(\d+)_chain(\d+)_frames(\d+)\.npy$"),
    "actual": re.compile(r"^actual_(.+?)_frame(\d+)_chain(\d+)_frames(\d+)\.npy$"),
}


def parse_entry(path, kind):
    name = path.name

    m = LEGACY_RE[kind].match(name)
    if m:
        frame = int(m.group(1))
        pdb_name = m.group(2)
        chain = int(m.group(3))
        return frame, chain, pdb_name, "legacy"

    m = MODERN_RE[kind].match(name)
    if m:
        pdb_name = m.group(1)
        frame = int(m.group(2))
        chain = int(m.group(3))
        return frame, chain, pdb_name, "modern"

    return None


def load_maps(input_dir, pdb_filter=None):
    pred_map = {}
    actual_map = {}

    for path in Path(input_dir).glob("*.npy"):
        pred_entry = parse_entry(path, "pred")
        if pred_entry is not None:
            frame, chain, pdb_name, _style = pred_entry
            if pdb_filter is None or pdb_name == pdb_filter:
                pred_map[(frame, chain)] = path
            continue

        actual_entry = parse_entry(path, "actual")
        if actual_entry is not None:
            frame, chain, pdb_name, _style = actual_entry
            if pdb_filter is None or pdb_name == pdb_filter:
                actual_map[(frame, chain)] = path

    return pred_map, actual_map


def safe_2d(arr):
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Expected 1D or 2D array, got shape {arr.shape}")


def compute_rmsd_rowwise(actual, pred):
    if actual.shape != pred.shape:
        raise ValueError(f"Shape mismatch: actual {actual.shape}, pred {pred.shape}")
    if actual.shape[1] % 3 != 0:
        raise ValueError(f"Coordinate width must be divisible by 3, got {actual.shape[1]}")

    values = []
    for idx in range(actual.shape[0]):
        a = actual[idx].reshape(1, -1, 3)
        p = pred[idx].reshape(-1, 3)
        if rmsd_traj is not None:
            values.append(float(rmsd_traj(a, p)))
        else:
            diff = actual[idx] - pred[idx]
            values.append(float(np.sqrt(np.mean(np.square(diff)))))
    return np.asarray(values, dtype=np.float64)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Attach per-chain predictions/actuals into full coordinates and compute RMSD."
    )
    parser.add_argument(
        "--input-dir",
        default=".",
        help="Directory containing chain-level .npy files from reverse scaling.",
    )
    parser.add_argument(
        "--pdb-name",
        default=None,
        help="Optional PDB tag filter (e.g., IgE). If omitted, all matching files are used.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Output prefix. Default: full_<pdb or merged>",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable progress logging.",
    )
    parser.add_argument(
        "--pred-only",
        action="store_true",
        help="Attach prediction chains only (skip actual and RMSD).",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()

    pred_map, actual_map = load_maps(args.input_dir, pdb_filter=args.pdb_name)
    shared_keys = sorted(set(pred_map.keys()) & set(actual_map.keys()))
    pred_only_keys = sorted(pred_map.keys())

    if args.pred_only:
        keys_to_use = pred_only_keys
        if not keys_to_use:
            raise SystemExit("No prediction chain files found for pred-only mode.")
    else:
        keys_to_use = shared_keys
        if not keys_to_use:
            raise SystemExit(
                "No matching pred/actual chain files found. "
                "Use --pred-only to attach predictions without labels."
            )

    frames = sorted({frame for frame, _chain in keys_to_use})
    all_pred_rows = []
    all_actual_rows = [] if not args.pred_only else None
    chains_seen = set()

    for frame in frames:
        chains = sorted(chain for (f, chain) in keys_to_use if f == frame)
        if not chains:
            continue
        chains_seen.update(chains)

        pred_parts = [safe_2d(np.load(pred_map[(frame, chain)])) for chain in chains]
        act_parts = (
            [safe_2d(np.load(actual_map[(frame, chain)])) for chain in chains]
            if not args.pred_only
            else None
        )

        rows = pred_parts[0].shape[0]
        if any(p.shape[0] != rows for p in pred_parts):
            raise ValueError(f"Inconsistent row counts across chains for frame {frame}")
        if not args.pred_only:
            if any(a.shape[0] != rows for a in act_parts):
                raise ValueError(f"Inconsistent actual row counts across chains for frame {frame}")
            if any(p.shape[0] != a.shape[0] for p, a in zip(pred_parts, act_parts)):
                raise ValueError(f"Pred/actual row count mismatch for frame {frame}")

        frame_pred = np.concatenate(pred_parts, axis=1)
        all_pred_rows.append(frame_pred)
        if not args.pred_only:
            frame_actual = np.concatenate(act_parts, axis=1)
            all_actual_rows.append(frame_actual)

    multi_pred = np.concatenate(all_pred_rows, axis=0)
    multi_act = np.concatenate(all_actual_rows, axis=0) if not args.pred_only else None
    rmsd_values = compute_rmsd_rowwise(multi_act, multi_pred) if not args.pred_only else None

    prefix = args.output_prefix
    if prefix is None:
        prefix = f"full_{args.pdb_name}" if args.pdb_name else "full_merged"

    np.save(f"{prefix}_prediction.npy", multi_pred)
    if not args.pred_only:
        np.save(f"{prefix}_actual.npy", multi_act)
        np.save(f"{prefix}_rmsd.npy", rmsd_values)

    if args.verbose:
        print(f"frames={len(frames)} chains={len(chains_seen)}")
        print(f"prediction shape: {multi_pred.shape}")
        if not args.pred_only:
            print(f"actual shape: {multi_act.shape}")
            print(f"rmsd shape: {rmsd_values.shape}")
            print(
                f"rmsd stats: mean={rmsd_values.mean():.6f} "
                f"min={rmsd_values.min():.6f} max={rmsd_values.max():.6f}"
            )


if __name__ == "__main__":
    main()
