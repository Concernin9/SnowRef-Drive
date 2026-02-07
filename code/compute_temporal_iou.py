import os
import re
from pathlib import Path

import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Config
# =========================
MASK_DIR = r"XXXXXX"
OUT_DIR  = r"XXXXXX"

N_FRAMES = 6  # frames per clip: 0..5
PATTERN = re.compile(r"^mask_(\d{6})_(\d)\.png$", re.IGNORECASE)

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# =========================
# Utilities
# =========================
def load_mask_as_bool(png_path: str) -> np.ndarray:
    """
    Load a PNG mask and convert to boolean foreground.
    Assumption: background=0, foreground>0 (common for binary masks).
    """
    img = Image.open(png_path).convert("L")  # grayscale
    arr = np.array(img, dtype=np.uint8)
    return arr > 0  # foreground

def iou_bool(a: np.ndarray, b: np.ndarray) -> float:
    """
    IoU for boolean masks a and b.
    If union is 0 (both empty), return 1.0 (perfect match).
    """
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(inter) / float(union)

def index_masks(mask_dir: str):
    """
    Build mapping: {clip_id_str: {frame_idx: filepath}}
    """
    mapping = {}
    for fn in os.listdir(mask_dir):
        m = PATTERN.match(fn)
        if not m:
            continue
        clip_id = m.group(1)
        frame_i = int(m.group(2))
        mapping.setdefault(clip_id, {})[frame_i] = str(Path(mask_dir) / fn)
    return mapping

# =========================
# Main
# =========================
mapping = index_masks(MASK_DIR)
if not mapping:
    raise RuntimeError(f"No mask files matched pattern in: {MASK_DIR}")

rows_pair = []   # per (clip, t->t+1) IoU
rows_clip = []   # per clip summary

missing_clips = 0

for clip_id, frames in sorted(mapping.items()):
    # Ensure 0..5 exist
    missing = [i for i in range(N_FRAMES) if i not in frames]
    if missing:
        missing_clips += 1
        continue

    # Load all 6 masks
    masks = [load_mask_as_bool(frames[i]) for i in range(N_FRAMES)]

    # Check consistent shape
    shapes = {m.shape for m in masks}
    if len(shapes) != 1:
        # Skip or handle resize; here we skip to avoid wrong IoU.
        missing_clips += 1
        continue

    # Compute adjacent IoUs
    ious = []
    for t in range(N_FRAMES - 1):
        v = iou_bool(masks[t], masks[t + 1])
        ious.append(v)
        rows_pair.append({
            "clip_id": clip_id,
            "pair": f"{t}-{t+1}",
            "iou": v
        })

    rows_clip.append({
        "clip_id": clip_id,
        "iou_mean_adjacent": float(np.mean(ious)),
        "iou_median_adjacent": float(np.median(ious)),
        "iou_min_adjacent": float(np.min(ious)),
        "iou_max_adjacent": float(np.max(ious)),
    })

df_pair = pd.DataFrame(rows_pair)
df_clip = pd.DataFrame(rows_clip)

if df_pair.empty:
    raise RuntimeError("No valid clips processed. Check filename pattern or missing frames.")

# Overall stats
overall = {
    "num_valid_clips": int(df_clip.shape[0]),
    "num_skipped_clips": int(missing_clips),
    "num_iou_pairs": int(df_pair.shape[0]),
    "iou_mean_all_pairs": float(df_pair["iou"].mean()),
    "iou_median_all_pairs": float(df_pair["iou"].median()),
    "iou_p05_all_pairs": float(df_pair["iou"].quantile(0.05)),
    "iou_p95_all_pairs": float(df_pair["iou"].quantile(0.95)),
}

df_overall = pd.DataFrame([overall])

# =========================
# Save Excel
# =========================
excel_path = str(Path(OUT_DIR) / "temporal_iou_results.xlsx")
with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    df_overall.to_excel(writer, sheet_name="overall", index=False)
    df_clip.to_excel(writer, sheet_name="per_clip", index=False)
    df_pair.to_excel(writer, sheet_name="per_pair", index=False)

# =========================
# Plot 1: Histogram of adjacent IoU
# =========================
plt.figure()
plt.hist(df_pair["iou"].values, bins=50)
plt.xlabel("IoU between adjacent frames")
plt.ylabel("Frequency")
plt.title("Temporal consistency: adjacent-frame IoU distribution")
hist_path = str(Path(OUT_DIR) / "iou_hist_adjacent.png")
plt.tight_layout()
plt.savefig(hist_path, dpi=300)
plt.close()

# =========================
# Plot 2: Boxplot (optional but nice)
# =========================
plt.figure()
plt.boxplot(df_pair["iou"].values, vert=True, showfliers=True)
plt.ylabel("IoU between adjacent frames")
plt.title("Temporal consistency: adjacent-frame IoU (boxplot)")
box_path = str(Path(OUT_DIR) / "iou_boxplot_adjacent.png")
plt.tight_layout()
plt.savefig(box_path, dpi=300)
plt.close()

print("Done.")
print(f"Excel saved to: {excel_path}")
print(f"Histogram saved to: {hist_path}")
print(f"Boxplot saved to: {box_path}")
print("Overall:", overall)
