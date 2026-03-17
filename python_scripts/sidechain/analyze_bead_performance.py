#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False

from bond_lookup import INT_TO_AA


def parse_args():
    p = argparse.ArgumentParser(
        description="Analyze run_model.py predictions by residue/bead and save CSV + plots."
    )
    p.add_argument("--feature", required=True, help="Feature file used for inference (.npy/.npz)")
    p.add_argument("--target", required=True, help="Target file (.npy/.npz)")
    p.add_argument("--mask", required=True, help="Mask file (.npy/.npz)")
    p.add_argument("--pred", required=True, help="Prediction file from run_model.py (.npy/.npz)")
    p.add_argument("--outdir", default="bead_performance_report", help="Output directory")
    p.add_argument(
        "--residue-col",
        type=int,
        default=37,
        help="0-based residue id column in feature array (default: 37 => input feature column 38)",
    )
    p.add_argument(
        "--bead-col",
        type=int,
        default=36,
        help="0-based bead id column in feature array (default: 36)",
    )
    p.add_argument("--top-k", type=int, default=20, help="Top K worst groups/samples to save")
    p.add_argument(
        "--filter-residue-id",
        type=int,
        default=None,
        help="Optional filter identical to run_model.py: remove rows with rounded X[:, gate_col] == this id",
    )
    p.add_argument(
        "--gate-col",
        type=int,
        default=37,
        help="Gate column for optional filter (default: 37)",
    )
    return p.parse_args()


def load_arr(path):
    if path.endswith(".npz"):
        arr = np.load(path)
        if "arr" not in arr:
            raise ValueError(f"{path} is .npz but missing key 'arr'")
        return arr["arr"].astype(np.float32)
    return np.load(path).astype(np.float32)


def to_group_matrix(arr, name):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[1] == 15:
        return arr
    if arr.ndim == 2 and arr.shape[1] % 15 == 0:
        return arr.reshape(-1, 15).astype(np.float32)
    if arr.ndim == 1 and arr.size % 15 == 0:
        return arr.reshape(-1, 15).astype(np.float32)
    raise ValueError(f"{name} cannot be reshaped to (*,15). Got {arr.shape}")


def residue_name_from_id(rid):
    rid = int(rid)
    if rid in INT_TO_AA:
        return INT_TO_AA[rid]
    if (rid - 1) in INT_TO_AA:
        return f"{INT_TO_AA[rid - 1]} (offset?)"
    return "UNK"


def grouped_stats(keys, rmse, mae):
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    rows = []
    for i in range(uniq.shape[0]):
        idx = inv == i
        r = int(uniq[i, 0])
        b = int(uniq[i, 1])
        vals = rmse[idx]
        maes = mae[idx]
        rows.append(
            (
                r,
                residue_name_from_id(r),
                b,
                int(idx.sum()),
                float(np.mean(vals)),
                float(np.median(vals)),
                float(np.percentile(vals, 90)),
                float(np.mean(maes)),
            )
        )
    rows.sort(key=lambda x: x[4], reverse=True)
    return rows


def save_csv(path, header, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(x) for x in row) + "\n")


def aggregate_1d(key, rmse, mae, label_fn=None):
    uniq, inv = np.unique(key, return_inverse=True)
    rows = []
    for i, k in enumerate(uniq):
        idx = inv == i
        vals = rmse[idx]
        maes = mae[idx]
        label = label_fn(int(k)) if label_fn else ""
        rows.append(
            (
                int(k),
                label,
                int(idx.sum()),
                float(np.mean(vals)),
                float(np.median(vals)),
                float(np.percentile(vals, 90)),
                float(np.mean(maes)),
            )
        )
    rows.sort(key=lambda x: x[3], reverse=True)
    return rows


def plot_worst_groups(rows, top_k, out_png):
    if not HAS_MATPLOTLIB:
        return
    top = rows[:top_k]
    if not top:
        return
    labels = [f"{r[0]}:{r[1]}|b{r[2]}" for r in top]
    vals = [r[4] for r in top]
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(top)), vals)
    plt.xticks(range(len(top)), labels, rotation=70, ha="right")
    plt.ylabel("Mean RMSE")
    plt.title(f"Top {len(top)} Worst Residue-Bead Groups")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def plot_heatmap(rows, out_png):
    if not HAS_MATPLOTLIB:
        return
    if not rows:
        return
    residues = sorted(set(r[0] for r in rows))
    beads = sorted(set(r[2] for r in rows))
    r_to_i = {r: i for i, r in enumerate(residues)}
    b_to_i = {b: i for i, b in enumerate(beads)}
    mat = np.full((len(residues), len(beads)), np.nan, dtype=np.float32)
    for r in rows:
        mat[r_to_i[r[0]], b_to_i[r[2]]] = r[4]
    plt.figure(figsize=(8, 7))
    im = plt.imshow(mat, aspect="auto")
    plt.colorbar(im, label="Mean RMSE")
    plt.xticks(range(len(beads)), [str(b) for b in beads])
    ylabels = [f"{r}:{residue_name_from_id(r)}" for r in residues]
    plt.yticks(range(len(residues)), ylabels)
    plt.xlabel("Bead ID")
    plt.ylabel("Residue ID")
    plt.title("Residue-Bead Mean RMSE Heatmap")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def plot_sample_error_hist(rmse, out_png):
    if not HAS_MATPLOTLIB:
        return
    plt.figure(figsize=(8, 5))
    plt.hist(rmse, bins=60)
    plt.xlabel("Per-group RMSE")
    plt.ylabel("Count")
    plt.title("Distribution of Per-group RMSE")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    X = load_arr(args.feature)
    Y = to_group_matrix(load_arr(args.target), "target")
    M = to_group_matrix(load_arr(args.mask), "mask")
    P = to_group_matrix(load_arr(args.pred), "pred")

    if X.ndim != 2:
        raise ValueError(f"Expected 2D feature matrix, got {X.shape}")
    if args.residue_col < 0 or args.residue_col >= X.shape[1]:
        raise ValueError(f"residue-col={args.residue_col} out of range for X shape {X.shape}")
    if args.bead_col < 0 or args.bead_col >= X.shape[1]:
        raise ValueError(f"bead-col={args.bead_col} out of range for X shape {X.shape}")

    if args.filter_residue_id is not None:
        gate = np.rint(X[:, args.gate_col]).astype(np.int32)
        keep = gate != int(args.filter_residue_id)
        X = X[keep]
        if Y.shape[0] == keep.shape[0]:
            Y = Y[keep]
            M = M[keep]

    if np.any(M < 0):
        M = (M > 0).astype(np.float32)

    if not (X.shape[0] == Y.shape[0] == M.shape[0] == P.shape[0]):
        raise ValueError(
            "Row mismatch after loading."
            f" X={X.shape}, Y={Y.shape}, M={M.shape}, P={P.shape}."
            " If prediction was filtered, pass matching filtered inputs or --filter-residue-id."
        )

    valid_coords = np.sum(M, axis=1)
    valid = valid_coords > 0
    if not np.any(valid):
        raise ValueError("Mask marks no valid coordinates.")

    se = ((P - Y) ** 2) * M
    ae = np.abs(P - Y) * M
    mse = np.zeros(Y.shape[0], dtype=np.float32)
    mae = np.zeros(Y.shape[0], dtype=np.float32)
    mse[valid] = np.sum(se[valid], axis=1) / valid_coords[valid]
    mae[valid] = np.sum(ae[valid], axis=1) / valid_coords[valid]
    rmse = np.sqrt(mse)

    residue_id = np.rint(X[:, args.residue_col]).astype(np.int32)
    bead_id = np.rint(X[:, args.bead_col]).astype(np.int32)
    keys = np.stack([residue_id, bead_id], axis=1)

    group_rows = grouped_stats(keys, rmse, mae)
    residue_rows = aggregate_1d(residue_id, rmse, mae, residue_name_from_id)
    bead_rows = aggregate_1d(bead_id, rmse, mae, lambda x: f"bead_{x}")

    save_csv(
        outdir / "per_residue_bead.csv",
        ["residue_id", "residue_name", "bead_id", "n", "mean_rmse", "median_rmse", "p90_rmse", "mean_mae"],
        group_rows,
    )
    save_csv(
        outdir / "per_residue.csv",
        ["residue_id", "residue_name", "n", "mean_rmse", "median_rmse", "p90_rmse", "mean_mae"],
        residue_rows,
    )
    save_csv(
        outdir / "per_bead.csv",
        ["bead_id", "bead_name", "n", "mean_rmse", "median_rmse", "p90_rmse", "mean_mae"],
        bead_rows,
    )

    worst_idx = np.argsort(-rmse)[: args.top_k]
    worst_rows = []
    for i in worst_idx:
        rid = int(residue_id[i])
        worst_rows.append(
            (
                int(i),
                rid,
                residue_name_from_id(rid),
                int(bead_id[i]),
                float(rmse[i]),
                float(mae[i]),
            )
        )
    save_csv(
        outdir / "worst_samples.csv",
        ["row_idx", "residue_id", "residue_name", "bead_id", "rmse", "mae"],
        worst_rows,
    )

    plot_worst_groups(group_rows, args.top_k, outdir / "worst_residue_bead_bar.png")
    plot_heatmap(group_rows, outdir / "residue_bead_heatmap.png")
    plot_sample_error_hist(rmse, outdir / "rmse_hist.png")

    overall_txt = outdir / "summary.txt"
    with open(overall_txt, "w", encoding="utf-8") as f:
        f.write("Bead Performance Summary\n")
        f.write(f"rows={X.shape[0]}\n")
        f.write(f"overall_mean_rmse={float(np.mean(rmse)):.6f}\n")
        f.write(f"overall_median_rmse={float(np.median(rmse)):.6f}\n")
        f.write(f"overall_p90_rmse={float(np.percentile(rmse, 90)):.6f}\n")
        if group_rows:
            w = group_rows[0]
            f.write("worst_group_by_mean_rmse=")
            f.write(f"residue_id:{w[0]} residue:{w[1]} bead:{w[2]} mean_rmse:{w[4]:.6f} n:{w[3]}\n")

    print("Saved analysis report to:", outdir.resolve())
    print(" -", (outdir / "summary.txt").name)
    print(" -", (outdir / "per_residue_bead.csv").name)
    print(" -", (outdir / "per_residue.csv").name)
    print(" -", (outdir / "per_bead.csv").name)
    print(" -", (outdir / "worst_samples.csv").name)
    if HAS_MATPLOTLIB:
        print(" -", (outdir / "worst_residue_bead_bar.png").name)
        print(" -", (outdir / "residue_bead_heatmap.png").name)
        print(" -", (outdir / "rmse_hist.png").name)
    else:
        print(" - plots skipped (matplotlib not installed)")


if __name__ == "__main__":
    main()
