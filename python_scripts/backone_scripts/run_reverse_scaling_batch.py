import argparse
import concurrent.futures
import os
import subprocess
import sys
from pathlib import Path


def parse_chain_lengths(text):
    vals = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not vals or any(v <= 0 for v in vals):
        raise ValueError("Invalid chain lengths. Use comma-separated positive integers.")
    return vals


def parse_frames(text):
    text = text.strip()
    if "-" in text:
        start_s, end_s = text.split("-", 1)
        start = int(start_s)
        end = int(end_s)
        if end < start:
            raise ValueError("Frame range must be start-end with end >= start.")
        return list(range(start, end + 1))
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def build_cmd(args, frame, chain, expected_length):
    pred = args.pred_template.format(frame=frame, chain=chain, pdb=args.pdb_name)
    actual = args.actual_template.format(frame=frame, chain=chain, pdb=args.pdb_name)
    cmin = args.custom_min_template.format(frame=frame, chain=chain, pdb=args.pdb_name)
    crange = args.custom_range_template.format(frame=frame, chain=chain, pdb=args.pdb_name)

    files = [pred, cmin, crange]
    if not args.cg_only:
        files.append(actual)
    missing = [f for f in files if not Path(f).exists()]
    if missing:
        return None, missing

    actual_arg = "-" if args.cg_only else actual

    cmd = [
        sys.executable,
        args.reverse_script,
        pred,
        actual_arg,
        cmin,
        crange,
        str(chain),
        args.pdb_name,
        str(frame),
        str(expected_length),
    ]
    return cmd, None


def run_one(cmd, verbose):
    env = os.environ.copy()
    if verbose:
        env["RAMA_VERBOSE"] = "1"
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return proc.returncode, proc.stdout, proc.stderr


def build_arg_parser():
    default_reverse_script = str(Path(__file__).with_name("new_reverse_scaling.py"))
    parser = argparse.ArgumentParser(
        description="Run new_reverse_scaling.py for many frames/chains in parallel."
    )
    parser.add_argument("--pdb-name", required=True, help="PDB tag used in filename templates.")
    parser.add_argument(
        "--chain-lengths",
        required=True,
        help="Comma-separated residue counts per chain (e.g., 546,215,546,215,121).",
    )
    parser.add_argument(
        "--frames",
        default="0-399",
        help="Frame list or range, e.g. '0-399' or '0,1,5,7'.",
    )
    parser.add_argument(
        "--reverse-script",
        default=default_reverse_script,
        help="Path to new_reverse_scaling.py",
    )
    parser.add_argument(
        "--pred-template",
        default="RAMAPROIR_yhat_frame_{frame}_chain_{chain}.npy",
        help="Prediction filename template.",
    )
    parser.add_argument(
        "--actual-template",
        default="train_LAB_B{frame}_{pdb}_chain{chain}.npy",
        help="Actual/label filename template.",
    )
    parser.add_argument(
        "--custom-min-template",
        default="custom_min_B{frame}_{pdb}_chain{chain}.npy",
        help="custom_min filename template.",
    )
    parser.add_argument(
        "--custom-range-template",
        default="custom_range_B{frame}_{pdb}_chain{chain}.npy",
        help="custom_range filename template.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=8,
        help="Parallel workers.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-task logs.",
    )
    parser.add_argument(
        "--cg-only",
        action="store_true",
        help="Run reverse-scaling without actual/label arrays.",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    chain_lengths = parse_chain_lengths(args.chain_lengths)
    frames = parse_frames(args.frames)

    tasks = []
    skipped = []
    for frame in frames:
        for chain_idx, residues in enumerate(chain_lengths, start=1):
            expected_length = residues * 12
            cmd, missing = build_cmd(args, frame, chain_idx, expected_length)
            if cmd is None:
                skipped.append((frame, chain_idx, missing))
            else:
                tasks.append((frame, chain_idx, cmd))

    if not tasks:
        raise SystemExit("No runnable tasks found (all files missing).")

    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        fut_map = {
            ex.submit(run_one, cmd, args.verbose): (frame, chain)
            for frame, chain, cmd in tasks
        }
        for fut in concurrent.futures.as_completed(fut_map):
            frame, chain = fut_map[fut]
            rc, out, err = fut.result()
            if rc != 0:
                failures.append((frame, chain, rc, err, out))
            elif args.verbose and out.strip():
                print(f"[frame={frame} chain={chain}] {out.strip()}")

    if skipped:
        print(f"Skipped {len(skipped)} tasks due to missing files.")
        if args.verbose:
            for frame, chain, missing in skipped:
                print(f"  frame={frame} chain={chain} missing={missing}")

    if failures:
        print(f"Failed {len(failures)} tasks:")
        for frame, chain, rc, err, out in failures[:10]:
            print(f"  frame={frame} chain={chain} rc={rc}")
            if err.strip():
                print(f"    stderr: {err.strip()}")
            elif out.strip():
                print(f"    stdout: {out.strip()}")
        raise SystemExit(1)

    print(f"Completed {len(tasks)} reverse-scaling tasks successfully.")


if __name__ == "__main__":
    main()
