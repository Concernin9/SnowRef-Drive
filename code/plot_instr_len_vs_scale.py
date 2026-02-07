#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import math
import time
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm


# -----------------------------
# Utilities
# -----------------------------
def now_tag():
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())

def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().strip()

def tokenize_len(text: str) -> int:
    # robust tokenization: words + numbers (keeps lane-words)
    toks = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?", text.lower())
    return len(toks)

def relation_score(text: str) -> float:
    """
    Return a continuous score in [0, 1] based on presence/count of relation words.
    Used for color encoding (blue -> purple gradient).
    """
    t = text.lower()

    phrases = [
        "next to", "in front of", "ahead of", "behind", "between",
        "left lane", "right lane", "center lane",
        "to the left of", "to the right of",
        "near", "adjacent", "alongside", "following",
        "in the middle of", "at the intersection", "on the crosswalk",
    ]
    words = ["left", "right", "behind", "ahead", "between", "near", "adjacent"]

    hits = 0
    for p in phrases:
        if p in t:
            hits += 2  # phrase = strong relation signal

    toks = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?", t)
    toks_set = set(toks)
    for w in words:
        if w in toks_set:
            hits += 1

    # squash to [0,1]
    score = 1.0 - math.exp(-hits / 4.0)
    return float(max(0.0, min(1.0, score)))

def mask_fg_pixels(mask_path: str) -> int:
    """
    Count foreground pixels in a mask png.
    Treat >0 as foreground.
    """
    im = Image.open(mask_path).convert("L")
    arr = np.array(im)
    return int((arr > 0).sum())

def compute_clip_area_stats(mask_dir: str, clip_id: int, n_frames: int,
                            min_fg_pixels: int, max_empty_frames: int):
    """
    Returns:
      - ok (bool)
      - mean_area (float) over non-empty frames
      - empty_frames (int) under min_fg_pixels
      - areas (list[int]) fg pixels per frame
      - reason (str) if not ok
    """
    areas = []
    empty = 0

    for k in range(n_frames):
        p = os.path.join(mask_dir, f"mask_{clip_id:06d}_{k}.png")
        if not os.path.exists(p):
            return False, None, None, None, f"missing_mask_frame_{k}"

        fg = mask_fg_pixels(p)
        areas.append(fg)

        if fg < min_fg_pixels:
            empty += 1

    if empty > max_empty_frames:
        return False, None, empty, areas, "too_many_empty_frames"

    non_empty_areas = [a for a in areas if a >= min_fg_pixels]
    if len(non_empty_areas) == 0:
        return False, None, empty, areas, "all_empty"

    mean_area = float(np.mean(non_empty_areas))
    return True, mean_area, empty, areas, "ok"


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask_dir", type=str, default="XXXXXX")
    parser.add_argument("--txt_dir", type=str, default="XXXXXX")
    parser.add_argument("--out_root", type=str, default="XXXXXX")

    parser.add_argument("--start_id", type=int, default=0)
    parser.add_argument("--num_clips", type=int, default=600)

    parser.add_argument("--n_frames", type=int, default=6)
    parser.add_argument("--min_fg_pixels", type=int, default=200)
    parser.add_argument("--max_empty_frames", type=int, default=3)

    # figure layout (2:1 rectangle by default)
    parser.add_argument("--fig_w", type=float, default=10.0)
    parser.add_argument("--fig_h", type=float, default=4.0)

    # plot controls
    parser.add_argument("--x_jitter", type=float, default=0.54,
                        help="Uniform jitter amplitude added to x for visualization, e.g., 0.25 -> x ± 0.25")
    parser.add_argument("--alpha", type=float, default=0.20,
                        help="Scatter alpha (transparency)")
    parser.add_argument("--point_size", type=float, default=6,
                        help="Scatter point size")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for reproducible jitter")

    args = parser.parse_args()

    run_dir = os.path.join(args.out_root, f"run_{now_tag()}")
    ensure_dir(run_dir)

    # custom colormap (as specified)
    cmap = LinearSegmentedColormap.from_list(
        "blue_purple_grad",
        ["#A8D8FF", "#C9B6FF", "#7B61FF"],
        N=256
    )

    rows = []
    skipped = {
        "missing_text": 0,
        "missing_mask": 0,
        "too_many_empty_frames": 0,
        "all_empty": 0,
        "other": 0,
    }

    clip_ids = range(args.start_id, args.start_id + args.num_clips)

    for cid in tqdm(clip_ids, desc="Scanning clips", ncols=100):
        # instruction file: prefer txt_{id}.txt, fallback to txt_{id}_0.txt
        txt_main = os.path.join(args.txt_dir, f"txt_{cid:06d}.txt")
        txt_fallback = os.path.join(args.txt_dir, f"txt_{cid:06d}_0.txt")

        if os.path.exists(txt_main):
            instr = read_text(txt_main)
            instr_path = txt_main
        elif os.path.exists(txt_fallback):
            instr = read_text(txt_fallback)
            instr_path = txt_fallback
        else:
            skipped["missing_text"] += 1
            continue

        ok, mean_area, empty_frames, areas, reason = compute_clip_area_stats(
            args.mask_dir, cid, args.n_frames, args.min_fg_pixels, args.max_empty_frames
        )

        if not ok:
            if reason.startswith("missing_mask_frame"):
                skipped["missing_mask"] += 1
            elif reason == "too_many_empty_frames":
                skipped["too_many_empty_frames"] += 1
            elif reason == "all_empty":
                skipped["all_empty"] += 1
            else:
                skipped["other"] += 1
            continue

        tok_len = tokenize_len(instr)
        rel = relation_score(instr)

        rows.append({
            "clip_id": cid,
            "instruction_path": instr_path,
            "instruction": instr,
            "token_len": tok_len,
            "relation_score": rel,
            "mean_area_nonempty": mean_area,
            "log10_mean_area_plus1": float(np.log10(mean_area + 1.0)),
            "empty_frames": int(empty_frames),
            "areas_per_frame": json.dumps(areas),
        })

    # Save CSV
    csv_path = os.path.join(run_dir, "data.csv")
    if rows:
        import csv
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    # Summary
    summary = {
        "run_dir": run_dir,
        "mask_dir": args.mask_dir,
        "txt_dir": args.txt_dir,
        "start_id": args.start_id,
        "num_clips_requested": args.num_clips,
        "num_rows_kept": len(rows),
        "skipped": skipped,
        "params": {
            "n_frames": args.n_frames,
            "min_fg_pixels": args.min_fg_pixels,
            "max_empty_frames": args.max_empty_frames,
            "x_jitter": args.x_jitter,
            "alpha": args.alpha,
            "point_size": args.point_size,
            "seed": args.seed,
        }
    }

    if rows:
        tok = np.array([r["token_len"] for r in rows], dtype=np.float32)
        loga = np.array([r["log10_mean_area_plus1"] for r in rows], dtype=np.float32)
        rels = np.array([r["relation_score"] for r in rows], dtype=np.float32)
        summary["stats"] = {
            "token_len_mean": float(tok.mean()),
            "token_len_median": float(np.median(tok)),
            "log_area_mean": float(loga.mean()),
            "log_area_median": float(np.median(loga)),
            "relation_score_mean": float(rels.mean()),
            "relation_score_nonzero_ratio": float((rels > 1e-6).mean()),
        }

    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Plot
    if rows:
        x = np.array([r["token_len"] for r in rows], dtype=np.float32)
        y = np.array([r["log10_mean_area_plus1"] for r in rows], dtype=np.float32)
        c = np.array([r["relation_score"] for r in rows], dtype=np.float32)

        # jitter only for visualization
        rng = np.random.default_rng(args.seed)
        if args.x_jitter > 0:
            x_plot = x + rng.uniform(-args.x_jitter, args.x_jitter, size=x.shape).astype(np.float32)
        else:
            x_plot = x

        plt.figure(figsize=(args.fig_w, args.fig_h))
        sc = plt.scatter(
            x_plot, y,
            c=c, cmap=cmap,
            s=args.point_size,
            alpha=args.alpha,
            edgecolors="none"
        )

        plt.xlabel("Instruction token length")
        plt.ylabel(r"$\log_{10}(\mathrm{mean\ mask\ area}+1)$")
        plt.title("Instruction Complexity vs Target Scale (SnowRef-Drive subset)")

        cb = plt.colorbar(sc)
        cb.set_label("Relation-word score (0 = none, 1 = strong)")

        plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
        plt.tight_layout()

        png_path = os.path.join(run_dir, "scatter.png")
        pdf_path = os.path.join(run_dir, "scatter.pdf")
        plt.savefig(png_path, dpi=300)
        plt.savefig(pdf_path)
        plt.close()

    print("DONE.")
    print(f"Run dir: {run_dir}")
    print(f"Kept rows: {len(rows)}")
    if rows:
        print(f"Saved: {csv_path}")
    print(f"Saved: {os.path.join(run_dir, 'summary.json')}")
    if rows:
        print(f"Saved: {os.path.join(run_dir, 'scatter.png')}")
        print(f"Saved: {os.path.join(run_dir, 'scatter.pdf')}")


if __name__ == "__main__":
    main()
