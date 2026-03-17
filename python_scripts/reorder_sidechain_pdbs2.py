#!/usr/bin/env python3
"""
Reorder sidechain-only PDB atom lines to match training ATOM_ORDER.

Supported input modes:
  - positional file paths
  - --input-dir (recursively scans --pattern)
  - --input-glob (one or more glob patterns)
  - --file-list (text file with one path per line)

By default outputs are written under --output-dir, preserving relative paths from
common input root. Use --in-place to overwrite source files.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import glob
import os
from pathlib import Path
from typing import Dict, List, Tuple

# Kept aligned with bond_lookup.py ATOM_ORDER
ATOM_ORDER: Dict[str, Dict[int, List[str]]] ={'LYS': {0: ['CB', 'CG', 'CD', 'CE', 'NZ']},
 'ALA': {0: ['CB']},
 'CYS': {0: ['CB', 'SG']},
 'GLN': {0: ['CB', 'CG', 'CD', 'OE1', 'NE2']},
 'VAL': {0: ['CB', 'CG1', 'CG2']},
 'ASN': {0: ['CB', 'CG', 'OD1', 'ND2']},
 'LEU': {0: ['CB', 'CG', 'CD1', 'CD2']},
 'THR': {0: ['CB', 'CG2', 'OG1']},
 'PHE': {0: ['CB', 'CG', 'CD1', 'CE1', 'CZ', 'CE2', 'CD2']},
 'SER': {0: ['CB', 'OG']},
 'PRO': {0: ['CD', 'CG', 'CB']},
 'TYR': {0: ['CB', 'CG', 'CD1', 'CE1', 'CZ', 'OH', 'CE2', 'CD2']},
 'HIS': {0: ['CB', 'CG', 'ND1', 'CE1', 'NE2', 'CD2']},
 'ARG': {0: ['CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2']},
 'TRP': {0: ['CB','CG','CD1','NE1','CE2','CZ2','CH2','CZ3','CE3','CD2']},
 'ILE': {0: ['CB', 'CG2', 'CG1', 'CD']},
 'GLU': {0: ['CB', 'CG', 'CD', 'OE1', 'OE2']},
 'ASP': {0: ['CB', 'CG', 'OD1', 'OD2']},
 'MET': {0: ['CB', 'CG', 'SD', 'CE']}}

RES_ALIASES = {
    "HSD": "HIS",
    "HSE": "HIS",
    "HSP": "HIS",
    "HID": "HIS",
    "HIE": "HIS",
    "HIP": "HIS",
    "MSE": "MET",
}


def flatten_atom_order(resname: str) -> List[str]:
    name = RES_ALIASES.get(resname, resname)
    per_bead = ATOM_ORDER.get(name)
    if per_bead is None:
        return []
    ordered: List[str] = []
    for bead in sorted(per_bead):
        ordered.extend(per_bead[bead])
    return ordered


def is_atom_line(line: str) -> bool:
    rec = line[0:6]
    return rec.startswith("ATOM") or rec.startswith("HETATM")


def atom_name(line: str) -> str:
    return line[12:16].strip()


def residue_key(line: str) -> Tuple[str, str, str, str]:
    # chain, residue id, insertion code, residue name
    return (line[21:22], line[22:26], line[26:27], line[17:20].strip())


def reorder_residue_block(block: List[str], strict: bool = False) -> Tuple[List[str], bool, List[str]]:
    if not block:
        return block, False, []

    resname = residue_key(block[0])[3]
    expected = flatten_atom_order(resname)
    if not expected:
        return block, False, []

    expected_set = set(expected)
    buckets: Dict[str, List[str]] = {a: [] for a in expected}
    extras: List[str] = []

    for ln in block:
        a = atom_name(ln)
        if a in expected_set:
            buckets[a].append(ln)
        else:
            extras.append(ln)

    missing = [a for a in expected if len(buckets[a]) == 0]
    unknown = [atom_name(ln) for ln in extras]

    if strict and (missing or unknown):
        rid = f"{residue_key(block[0])[0]}:{residue_key(block[0])[1].strip()}{residue_key(block[0])[2].strip()}"
        msg = []
        if missing:
            msg.append(f"missing={missing}")
        if unknown:
            msg.append(f"unknown={unknown}")
        raise ValueError(f"Strict check failed for {resname} {rid}: " + ", ".join(msg))

    out: List[str] = []
    for a in expected:
        out.extend(buckets[a])
    out.extend(extras)

    changed = out != block

    notes: List[str] = []
    if missing:
        notes.append(f"{resname}: missing expected atoms {missing}")
    if unknown:
        notes.append(f"{resname}: unknown atoms kept at end {unknown}")

    return out, changed, notes


def renumber_serials(lines: List[str]) -> List[str]:
    out: List[str] = []
    serial = 1
    for ln in lines:
        if is_atom_line(ln):
            if len(ln) < 11:
                out.append(ln)
            else:
                out.append(f"{ln[:6]}{serial:5d}{ln[11:]}")
                serial += 1
        else:
            out.append(ln)
    return out


def reorder_pdb_text(lines: List[str], strict: bool = False, renumber: bool = False) -> Tuple[List[str], int, int, List[str]]:
    out: List[str] = []
    block: List[str] = []
    block_key: Tuple[str, str, str, str] | None = None

    residues_seen = 0
    residues_changed = 0
    notes: List[str] = []

    def flush_block() -> None:
        nonlocal block, block_key, residues_seen, residues_changed, out, notes
        if not block:
            return
        reordered, changed, n = reorder_residue_block(block, strict=strict)
        out.extend(reordered)
        residues_seen += 1
        if changed:
            residues_changed += 1
        notes.extend(n)
        block = []
        block_key = None

    for ln in lines:
        if is_atom_line(ln):
            key = residue_key(ln)
            if block_key is None:
                block_key = key
                block = [ln]
            elif key == block_key:
                block.append(ln)
            else:
                flush_block()
                block_key = key
                block = [ln]
        else:
            flush_block()
            out.append(ln)

    flush_block()

    if renumber:
        out = renumber_serials(out)

    return out, residues_seen, residues_changed, notes


def discover_files(args: argparse.Namespace) -> List[Path]:
    seen: Dict[str, Path] = {}

    def add_path(p: Path) -> None:
        rp = p.expanduser().resolve()
        if rp.is_file():
            seen[str(rp)] = rp

    for s in args.inputs:
        add_path(Path(s))

    for pat in args.input_glob:
        for s in glob.glob(pat, recursive=True):
            add_path(Path(s))

    for d in args.input_dir:
        root = Path(d).expanduser().resolve()
        if not root.is_dir():
            continue
        for p in root.rglob(args.pattern):
            add_path(p)

    for fl in args.file_list:
        path = Path(fl).expanduser().resolve()
        if not path.is_file():
            continue
        for raw in path.read_text().splitlines():
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            add_path(Path(raw))

    files = sorted(seen.values())
    if not files:
        raise ValueError("No input PDB files found. Provide paths, --input-dir, --input-glob, or --file-list.")
    return files


def output_path_for(src: Path, output_dir: Path, common_root: Path) -> Path:
    rel = src.relative_to(common_root)
    return output_dir / rel


def process_one(
    src: Path,
    dst: Path,
    strict: bool,
    renumber: bool,
    overwrite: bool,
) -> Tuple[str, int, int, int, List[str], str | None]:
    try:
        text = src.read_text().splitlines(keepends=True)
        out, residues_seen, residues_changed, notes = reorder_pdb_text(text, strict=strict, renumber=renumber)

        if (not overwrite) and dst.exists() and src != dst:
            return str(src), residues_seen, residues_changed, len(out), notes, f"Output exists: {dst}"

        dst.parent.mkdir(parents=True, exist_ok=True)
        if src == dst:
            tmp = dst.with_suffix(dst.suffix + ".tmp_reordered")
            tmp.write_text("".join(out))
            os.replace(tmp, dst)
        else:
            dst.write_text("".join(out))

        return str(src), residues_seen, residues_changed, len(out), notes, None
    except Exception as exc:  # noqa: BLE001
        return str(src), 0, 0, 0, [], str(exc)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reorder sidechain-only PDB atoms to training order.")
    p.add_argument("inputs", nargs="*", help="Input PDB file paths.")
    p.add_argument("--input-dir", action="append", default=[], help="Directory to recursively scan for PDBs.")
    p.add_argument("--pattern", default="*.pdb", help="Pattern used with --input-dir (default: *.pdb).")
    p.add_argument("--input-glob", action="append", default=[], help="Glob pattern(s), supports **.")
    p.add_argument("--file-list", action="append", default=[], help="Text file(s) with one input path per line.")

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output-dir", type=Path, help="Write reordered files under this directory.")
    mode.add_argument("--in-place", action="store_true", help="Overwrite each input file in place.")

    p.add_argument("--workers", type=int, default=1, help="Parallel workers (default: 1).")
    p.add_argument("--strict", action="store_true", help="Fail a residue if expected atoms are missing or unknown atoms exist.")
    p.add_argument("--renumber-serial", action="store_true", help="Renumber ATOM/HETATM serials sequentially in output.")
    p.add_argument("--overwrite", action="store_true", help="Allow overwriting existing files in --output-dir mode.")
    p.add_argument("--report", type=Path, help="Optional report file with warnings/errors.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    files = discover_files(args)

    if args.in_place:
        pairs = [(f, f) for f in files]
    else:
        out_dir = args.output_dir.expanduser().resolve()
        common_root = Path(os.path.commonpath([str(f) for f in files]))
        pairs = [(f, output_path_for(f, out_dir, common_root)) for f in files]

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = [
            ex.submit(
                process_one,
                src,
                dst,
                args.strict,
                args.renumber_serial,
                args.overwrite,
            )
            for src, dst in pairs
        ]
        for fut in concurrent.futures.as_completed(futs):
            results.append(fut.result())

    results.sort(key=lambda x: x[0])

    ok = 0
    err = 0
    residues = 0
    changed = 0
    report_lines: List[str] = []

    for src, n_res, n_changed, n_lines, notes, error in results:
        if error:
            err += 1
            report_lines.append(f"ERROR {src}: {error}")
            continue
        ok += 1
        residues += n_res
        changed += n_changed
        for n in notes:
            report_lines.append(f"WARN  {src}: {n}")

    print(f"Processed files: {len(results)}")
    print(f"Succeeded: {ok}")
    print(f"Failed: {err}")
    print(f"Residues seen: {residues}")
    print(f"Residues reordered: {changed}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text("\n".join(report_lines) + ("\n" if report_lines else ""))
        print(f"Report written: {args.report}")
    elif report_lines:
        print("Warnings:")
        for line in report_lines[:30]:
            print(line)
        if len(report_lines) > 30:
            print(f"... {len(report_lines) - 30} more warnings (use --report to save all)")

    if err > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
