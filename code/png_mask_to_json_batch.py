import os
import re
import json
import argparse
import numpy as np
import cv2
from tqdm import tqdm

PNG_RE = re.compile(r"^mask_(\d{6})_(\d)\.png$", re.IGNORECASE)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# -------------------------
# Lossless RLE encoder (COCO-style uncompressed)
# -------------------------
def mask_to_rle(binary_mask: np.ndarray):
    h, w = binary_mask.shape
    flat = binary_mask.flatten(order="F").astype(np.uint8)

    counts = []
    prev = 0
    run_len = 0

    for v in flat:
        if v == prev:
            run_len += 1
        else:
            counts.append(run_len)
            run_len = 1
            prev = v

    counts.append(run_len)

    return {
        "order": "F",
        "size": [int(h), int(w)],
        "counts": counts
    }

# -------------------------
# Polygon extractor
# -------------------------
def contours_to_polygons(binary_mask: np.ndarray):
    mask = (binary_mask * 255).astype(np.uint8)

    contours, hierarchy = cv2.findContours(
        mask,
        cv2.RETR_CCOMP,         # keep holes
        cv2.CHAIN_APPROX_NONE   # max fidelity
    )

    polygons = []

    if hierarchy is None:
        return polygons

    hierarchy = hierarchy[0]

    for i, cnt in enumerate(contours):
        if cnt is None or len(cnt) < 3:
            continue

        parent = hierarchy[i][3]
        is_hole = parent != -1

        pts = cnt.reshape(-1, 2).tolist()
        area = int(abs(cv2.contourArea(cnt)))

        polygons.append({
            "points": pts,
            "is_hole": bool(is_hole),
            "area": area
        })

    return polygons

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask_dir", type=str, required=True, help="folder with mask_XXXXXX_Y.png")
    ap.add_argument("--json_dir", type=str, required=True, help="output folder for json")
    ap.add_argument("--thr", type=int, default=0, help="foreground threshold (img > thr)")
    ap.add_argument("--skip_existing", action="store_true", help="skip if json already exists")
    args = ap.parse_args()

    ensure_dir(args.json_dir)

    files = sorted([f for f in os.listdir(args.mask_dir) if PNG_RE.match(f)])
    if len(files) == 0:
        print("No mask png found.")
        return

    total = 0
    empty_cnt = 0
    broken_cnt = 0
    skipped_cnt = 0

    pbar = tqdm(files, desc="PNG -> JSON", unit="file", dynamic_ncols=True)

    for fname in pbar:
        png_path = os.path.join(args.mask_dir, fname)
        json_name = fname[:-4] + ".json"  # replace .png
        json_path = os.path.join(args.json_dir, json_name)

        if args.skip_existing and os.path.exists(json_path):
            skipped_cnt += 1
            pbar.set_postfix_str(f"done={total} empty={empty_cnt} broken={broken_cnt} skipped={skipped_cnt}")
            continue

        img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            broken_cnt += 1
            pbar.set_postfix_str(f"done={total} empty={empty_cnt} broken={broken_cnt} skipped={skipped_cnt}")
            continue

        h, w = img.shape
        binary = (img > args.thr).astype(np.uint8)
        has_mask = bool(binary.any())

        polygons = contours_to_polygons(binary)
        rle = mask_to_rle(binary)

        data = {
            "version": "snowref-mask-v1",
            "image_size": [int(h), int(w)],
            "format": "binary",
            "has_mask": has_mask,
            "polygons": polygons,
            "rle": rle
        }

        if not has_mask:
            empty_cnt += 1

        # ---- Pretty JSON: multi-line with indentation ----
        # ensure_ascii=False keeps it human-friendly if later you add Chinese fields
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        total += 1
        pbar.set_postfix_str(f"done={total} empty={empty_cnt} broken={broken_cnt} skipped={skipped_cnt}")

    print("================================")
    print("Finished")
    print("Converted:", total)
    print("Empty masks:", empty_cnt)
    print("Broken files:", broken_cnt)
    print("Skipped existing:", skipped_cnt)

if __name__ == "__main__":
    main()
