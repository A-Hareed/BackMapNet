import argparse
from collections import OrderedDict
from pathlib import Path

import numpy as np


DEFAULT_CHAIN_LENGTHS = [546, 215, 546, 215, 121]


def parse_chain_lengths(text):
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        raise ValueError("Empty chain lengths string.")
    lengths = [int(x) for x in parts]
    if any(x <= 0 for x in lengths):
        raise ValueError("All chain lengths must be positive integers.")
    return lengths


def load_chain_lengths_from_file(path):
    data = Path(path).read_text(encoding="utf-8").strip()
    return parse_chain_lengths(data)


def infer_chain_lengths_from_pdb(pdb_path):
    """
    Infer chain residue counts from a PDB file using either:
    1) chain label changes, or
    2) residue-number restarts within the same chain label.

    Works for both CG (BB beads) and AA (CA atoms):
    - If BB atoms are present, BB is used.
    - Otherwise CA is used.
    """
    bb_records = []
    ca_records = []

    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            if len(line) < 27:
                continue

            atom_name = line[12:16].strip()
            chain_id = line[21].strip() or "_"
            resseq_text = line[22:26].strip()
            if not resseq_text:
                continue
            try:
                resseq = int(resseq_text)
            except ValueError:
                continue
            ins_code = line[26].strip()
            rec = (chain_id, resseq, ins_code)
            if atom_name == "BB":
                bb_records.append(rec)
            elif atom_name == "CA":
                ca_records.append(rec)

    records = bb_records if bb_records else ca_records
    if not records:
        raise ValueError(f"No BB/CA atoms found in PDB: {pdb_path}")

    lengths = []
    current_count = 0
    last_chain = None
    last_resseq = None
    last_reskey = None
    seen_residues = set()

    for chain_id, resseq, ins_code in records:
        reskey = (resseq, ins_code)

        boundary = False
        if last_chain is None:
            boundary = True
        elif chain_id != last_chain:
            boundary = True
        else:
            # Same chain label: detect restart/reuse of residue numbering.
            if resseq < last_resseq:
                boundary = True
            elif reskey in seen_residues and reskey != last_reskey:
                boundary = True

        if boundary:
            if current_count > 0:
                lengths.append(current_count)
            current_count = 0
            last_reskey = None
            seen_residues = set()

        if reskey != last_reskey:
            current_count += 1
            seen_residues.add(reskey)
            last_reskey = reskey

        last_chain = chain_id
        last_resseq = resseq

    if current_count > 0:
        lengths.append(current_count)

    return lengths


def create_feature_set_fast(data, window_size, step_size):
    """
    Same ordering as the original implementation:
    window-major rows: all samples for window 0, then all samples for window 1, ...
    """
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D data, got shape {data.shape}")
    if data.shape[1] < window_size:
        raise ValueError(
            f"window_size={window_size} is larger than data width={data.shape[1]}"
        )

    all_windows = np.lib.stride_tricks.sliding_window_view(
        data, window_shape=window_size, axis=1
    )  # [samples, width-window+1, window]
    stepped = all_windows[:, ::step_size, :]  # [samples, num_windows, window]
    return np.transpose(stepped, (1, 0, 2)).reshape(-1, window_size)


def normalize_fragments_per_axis(fragments):
    """
    Per-window, per-axis normalization with +/-4 padding around min/max.
    """
    absolute_min = np.min(fragments, axis=1, keepdims=True)
    absolute_max = np.max(fragments, axis=1, keepdims=True)
    custom_min = absolute_min - 4.0
    custom_max = absolute_max + 4.0
    custom_range = custom_max - custom_min
    normalized_fragments = (fragments - custom_min) / custom_range
    return normalized_fragments, custom_min, custom_range


def split_by_chain(flat_array, chain_lengths, dims_per_residue):
    split_points = np.cumsum(np.asarray(chain_lengths) * dims_per_residue)[:-1]
    return np.split(flat_array, split_points.astype(int), axis=1)


def validate_lengths(feat_arr, target_arr, chain_lengths):
    expected_feat = int(sum(chain_lengths) * 3)
    expected_target = int(sum(chain_lengths) * 12)
    if feat_arr.shape[1] != expected_feat:
        raise ValueError(
            f"CG width mismatch: got {feat_arr.shape[1]}, expected {expected_feat} "
            f"from chain_lengths={chain_lengths}"
        )
    if target_arr.shape[1] != expected_target:
        raise ValueError(
            f"Target width mismatch: got {target_arr.shape[1]}, expected {expected_target} "
            f"from chain_lengths={chain_lengths}"
        )


def validate_feat_only(feat_arr, chain_lengths):
    expected_feat = int(sum(chain_lengths) * 3)
    if feat_arr.shape[1] != expected_feat:
        raise ValueError(
            f"CG width mismatch: got {feat_arr.shape[1]}, expected {expected_feat} "
            f"from chain_lengths={chain_lengths}"
        )


def resolve_chain_lengths(args, feat_arr, target_arr=None):
    if args.chain_lengths:
        chain_lengths = parse_chain_lengths(args.chain_lengths)
    elif args.chain_lengths_file:
        chain_lengths = load_chain_lengths_from_file(args.chain_lengths_file)
    elif args.chain_lengths_from_pdb:
        chain_lengths = infer_chain_lengths_from_pdb(args.chain_lengths_from_pdb)
    else:
        chain_lengths = DEFAULT_CHAIN_LENGTHS
    if target_arr is None:
        validate_feat_only(feat_arr, chain_lengths)
    else:
        validate_lengths(feat_arr, target_arr, chain_lengths)
    return chain_lengths


def save_chain_outputs(
    chain_feat,
    chain_target,
    frame_idx,
    pdb_name,
    chain_number,
    window_residues=32,
):
    cg_window = window_residues * 3

    batch_features = create_feature_set_fast(chain_feat, cg_window, 3).reshape(-1, 32, 3)

    array_cg, custom_min, custom_range = normalize_fragments_per_axis(batch_features)

    np.save(f"train_feat_B{frame_idx}_{pdb_name}_chain{chain_number}.npy", array_cg.reshape(-1, 32 * 3))
    np.save(f"custom_min_B{frame_idx}_{pdb_name}_chain{chain_number}.npy", custom_min)
    np.save(f"custom_range_B{frame_idx}_{pdb_name}_chain{chain_number}.npy", custom_range)

    aa_shape = None
    if chain_target is not None:
        aa_window = window_residues * 12
        batch_lab = create_feature_set_fast(chain_target, aa_window, 12).reshape(-1, 128, 3)
        array_aa = (batch_lab - custom_min) / custom_range
        np.save(f"train_LAB_B{frame_idx}_{pdb_name}_chain{chain_number}.npy", array_aa.reshape(-1, 128 * 3))
        aa_shape = array_aa.shape

    return {
        "chain": chain_number,
        "residues": chain_feat.shape[1] // 3,
        "windows": array_cg.shape[0],
        "cg_shape": array_cg.shape,
        "aa_shape": aa_shape,
    }


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Create per-frame, per-chain CG/AA sliding-window training arrays."
    )
    parser.add_argument("pdb_name", help="Name tag used in output filenames (e.g., IgE).")
    parser.add_argument("frame_idx", help="Frame index used in input/output filenames.")
    parser.add_argument(
        "--chain-lengths",
        help="Comma-separated residue counts per chain (e.g., 546,215,546,215,121).",
    )
    parser.add_argument(
        "--chain-lengths-file",
        help="Text file containing comma-separated residue counts per chain.",
    )
    parser.add_argument(
        "--chain-lengths-from-pdb",
        help="Infer chain lengths from this PDB file (BB if present, otherwise CA).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable progress logging.",
    )
    parser.add_argument(
        "--cg-only",
        action="store_true",
        help="Generate only train_feat/custom_* from CG (skip train_LAB outputs).",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()

    frame_token = str(args.frame_idx).upper()
    if frame_token == "ALL":
        feat_path = "cluster_ALL_CG.npy"
        feat_all = np.load(feat_path).astype(np.float32)
        target_all = None
        if not args.cg_only:
            target_path = "cluster_ALL.npy"
            target_all = np.load(target_path).astype(np.float32)
            if feat_all.shape[0] != target_all.shape[0]:
                raise ValueError(
                    f"Frame count mismatch in ALL arrays: feat={feat_all.shape[0]} target={target_all.shape[0]}"
                )

        chain_lengths = resolve_chain_lengths(args, feat_all, target_all)
        n_frames = feat_all.shape[0]
        frame_idx_out = 0

        # Aggregate all frames into a single file-set per chain.
        feat_splits = split_by_chain(feat_all, chain_lengths, dims_per_residue=3)
        target_splits = (
            None
            if target_all is None
            else split_by_chain(target_all, chain_lengths, dims_per_residue=12)
        )

        summaries = []
        for chain_idx, chain_feat in enumerate(feat_splits, start=1):
            chain_target = None if target_splits is None else target_splits[chain_idx - 1]
            summary = save_chain_outputs(
                chain_feat=chain_feat,
                chain_target=chain_target,
                frame_idx=frame_idx_out,
                pdb_name=args.pdb_name,
                chain_number=chain_idx,
                window_residues=32,
            )
            summaries.append(summary)

        if args.verbose:
            print(
                f"ALL mode complete for {args.pdb_name}: frames={n_frames}, "
                f"chain_lengths={chain_lengths}, output_token=B{frame_idx_out}"
            )
            for s in summaries:
                print(
                    f"  chain{s['chain']}: residues={s['residues']} windows={s['windows']} "
                    f"cg={s['cg_shape']} aa={s['aa_shape']}"
                )
    else:
        feat_path = f"cluster_{args.frame_idx}_CG.npy"
        feat_train = np.load(feat_path).astype(np.float32)
        target_train = None
        if not args.cg_only:
            target_path = f"cluster_{args.frame_idx}.npy"
            target_train = np.load(target_path).astype(np.float32)

        chain_lengths = resolve_chain_lengths(args, feat_train, target_train)

        feat_splits = split_by_chain(feat_train, chain_lengths, dims_per_residue=3)
        target_splits = (
            None
            if target_train is None
            else split_by_chain(target_train, chain_lengths, dims_per_residue=12)
        )

        summaries = []
        for chain_idx, chain_feat in enumerate(feat_splits, start=1):
            chain_target = None if target_splits is None else target_splits[chain_idx - 1]
            summary = save_chain_outputs(
                chain_feat=chain_feat,
                chain_target=chain_target,
                frame_idx=args.frame_idx,
                pdb_name=args.pdb_name,
                chain_number=chain_idx,
                window_residues=32,
            )
            summaries.append(summary)

        if args.verbose:
            print(f"Frame {args.frame_idx} ({args.pdb_name}) done.")
            print(f"Chain lengths: {chain_lengths}")
            for s in summaries:
                print(
                    f"  chain{s['chain']}: residues={s['residues']} windows={s['windows']} "
                    f"cg={s['cg_shape']} aa={s['aa_shape']}"
                )


if __name__ == "__main__":
    main()
