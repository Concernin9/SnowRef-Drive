import os
import re
import time
import argparse
from pathlib import Path

from PIL import Image, ImageEnhance
from tqdm import tqdm


# =========================
# 固定协议：6帧
# =========================
N_FRAMES = 6
IMG_RE = re.compile(r"^img_(\d{6})_(\d)\.jpg$", re.IGNORECASE)

# 认为“有效输出文件”的最小大小（防止断电产生 0KB/很小的坏图被误跳过）
VALID_OUT_MIN_BYTES = 50 * 1024  # 50KB


def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def index_images_only(img_dir: str):
    """
    只索引 img_dir：
    返回: {id_str: {frame_idx: filepath}}
    """
    folder_path = Path(img_dir)
    if not folder_path.exists():
        raise FileNotFoundError(f"img_dir not found: {img_dir}")

    mapping = {}
    for p in folder_path.iterdir():
        if not p.is_file():
            continue
        m = IMG_RE.match(p.name)
        if not m:
            continue
        idx = m.group(1)
        fr = int(m.group(2))
        mapping.setdefault(idx, {})
        mapping[idx][fr] = str(p)

    print(f"[INDEX] img: {img_dir} -> IDs: {len(mapping)}")
    return mapping


def mask_path_for(mask_dir: str, idx: str, fr: int) -> str:
    # mask 命名固定：mask_{xxxxxx}_{y}.png
    return str(Path(mask_dir) / f"mask_{idx}_{fr}.png")


# =========================================================
# ✅ 等比例放大 + 裁剪回原尺寸（可偏移）
# =========================================================
def zoom_crop_keep_size(img: Image.Image, zoom: float, shift_x: int, shift_y: int, resample) -> Image.Image:
    """
    把图等比例放大 zoom 倍，然后裁剪成原尺寸 (W,H)。
    裁剪窗口中心默认对齐中心，可通过 shift_x/shift_y 偏移（像素，基于原尺寸坐标）。
      shift_x > 0：裁剪窗口向右（更保留右边，裁掉左边更多）
      shift_x < 0：裁剪窗口向左（更保留左边，裁掉右边更多）→ 去右上水印通常用负数
      shift_y > 0：裁剪窗口向下（更保留下方，裁掉上方更多）→ 去上方水印通常用正数
    """
    if zoom <= 1.0:
        return img

    w, h = img.size
    new_w = int(round(w * zoom))
    new_h = int(round(h * zoom))

    # 先放大
    big = img.resize((new_w, new_h), resample=resample)

    # 在放大图上，裁剪一个 w*h 的窗口
    # 中心点：放大图中心 + shift（把 shift 按 zoom 映射到放大坐标）
    cx = new_w / 2.0 + shift_x * zoom
    cy = new_h / 2.0 + shift_y * zoom

    left = int(round(cx - w / 2.0))
    top  = int(round(cy - h / 2.0))

    # clamp，确保窗口不出界
    left = max(0, min(left, new_w - w))
    top  = max(0, min(top,  new_h - h))

    return big.crop((left, top, left + w, top + h))


def overlay_mask_on_image(
    img_rgb: Image.Image,
    mask_l: Image.Image,
    darken: float,
    color_rgb,
    alpha_value: int
):
    """
    输入已对齐尺寸的 img_rgb / mask_l
    输出：调暗的原图 + 红色mask叠加
    """
    # 1) 底图调暗
    enhancer = ImageEnhance.Brightness(img_rgb)
    dark_img = enhancer.enhance(darken)
    base = dark_img.convert("RGBA")

    # 2) mask 二值化
    m = mask_l.convert("L")
    if m.size != img_rgb.size:
        m = m.resize(img_rgb.size, resample=Image.NEAREST)
    bin_m = m.point(lambda p: 255 if p > 0 else 0)

    # 3) 彩色 overlay
    overlay = Image.new("RGBA", base.size, (color_rgb[0], color_rgb[1], color_rgb[2], 0))
    alpha = bin_m.point(lambda p: alpha_value if p > 0 else 0)
    overlay.putalpha(alpha)

    # 4) 合成
    out = Image.alpha_composite(base, overlay).convert("RGB")
    return out


def vstack(images, gap_y, bg_color):
    widths = [im.width for im in images]
    heights = [im.height for im in images]
    max_w = max(widths)
    total_h = sum(heights) + gap_y * (len(images) - 1)
    canvas = Image.new("RGB", (max_w, total_h), bg_color)

    y = 0
    for im in images:
        canvas.paste(im, (0, y))
        y += im.height + gap_y
    return canvas


def hstack(images, gap_x, bg_color):
    widths = [im.width for im in images]
    heights = [im.height for im in images]
    total_w = sum(widths) + gap_x * (len(images) - 1)
    max_h = max(heights)
    canvas = Image.new("RGB", (total_w, max_h), bg_color)

    x = 0
    for im in images:
        canvas.paste(im, (x, 0))
        x += im.width + gap_x
    return canvas


def save_jpeg_under_limit(img: Image.Image, out_path: str, max_size_mb: float, max_passes: int):
    """
    目标：尽量压到 <= max_size_mb
    最多尝试 max_passes 次（建议2），绝不死循环。
    """
    quality_passes = [
        [92, 80, 70, 60],
        [55, 45, 38, 32],
    ]
    quality_passes = quality_passes[:max(1, min(max_passes, len(quality_passes)))]

    last_q, last_s = 60, 999.0

    for qs in quality_passes:
        for q in qs:
            img.save(out_path, format="JPEG", quality=q, optimize=True)
            size_mb = os.path.getsize(out_path) / 1024 / 1024
            last_q, last_s = q, size_mb
            if size_mb <= max_size_mb:
                return last_q, last_s

    return last_q, last_s


def parse_args():
    ap = argparse.ArgumentParser(
        description="Make 2-column (raw | raw+mask) mosaics for 6-frame clips, with zoom-crop to remove watermark (resume supported)."
    )

    ap.add_argument("--img_dir", default=r"XXXXXX",
                    help="Folder of raw images: img_XXXXXX_y.jpg")
    ap.add_argument("--mask_dir", default=r"XXXXXX",
                    help="Folder of masks: mask_XXXXXX_y.png")

    ap.add_argument("--gap_row", type=int, default=15, help="Gap between frames within a column (small).")
    ap.add_argument("--gap_col", type=int, default=40, help="Gap between two columns (large).")

    ap.add_argument("--out_root", default="mosaic_results", help="Output root folder name under video/.")
    ap.add_argument("--prefix", default="mosaic", help="Output file prefix: prefix_XXXXXX.jpg")

    ap.add_argument("--darken", type=float, default=0.65, help="Base image darkening factor (0~1+).")
    ap.add_argument("--alpha", type=int, default=160, help="Mask overlay alpha (0~255).")
    ap.add_argument("--color", type=int, nargs=3, default=[255, 60, 60], help="Mask color RGB, e.g. 255 60 60")

    ap.add_argument("--max_mb", type=float, default=1.0, help="Max output JPEG size in MB.")
    ap.add_argument("--max_passes", type=int, default=2, help="Max compression passes (<=2 recommended).")

    # ✅ 新增：等比例放大裁剪
    ap.add_argument("--zoom", type=float, default=1.18,
                    help="Zoom factor (>1.0). 1.18 means enlarge 18%% then crop back to original size.")
    ap.add_argument("--shift_x", type=int, default=-180,
                    help="Crop window shift in pixels (original scale). Negative moves window left (good for removing top-right watermark).")
    ap.add_argument("--shift_y", type=int, default=60,
                    help="Crop window shift in pixels (original scale). Positive moves window down (good for removing top watermark).")

    ap.add_argument("--start_id", default=None, help="Optional: only process IDs >= start_id (6 digits).")
    ap.add_argument("--end_id", default=None, help="Optional: only process IDs <= end_id (6 digits).")
    ap.add_argument("--limit", type=int, default=None, help="Optional: limit number of mosaics for quick test.")
    ap.add_argument("--verbose_missing", action="store_true",
                    help="Print why a sample is skipped (missing frame/mask).")

    # ✅ 断点续跑机制
    ap.add_argument("--resume", action="store_true", default=True,
                    help="Skip already generated mosaic files if they look valid (default: on)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Force overwrite existing mosaic files")
    ap.add_argument("--min_valid_kb", type=int, default=50,
                    help="If resume is on, output file must be >= this KB to be considered valid (default: 50KB)")

    return ap.parse_args()


def main():
    args = parse_args()

    img_dir = Path(args.img_dir)
    mask_dir = Path(args.mask_dir)
    if not img_dir.exists():
        raise FileNotFoundError(f"img_dir not found: {img_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"mask_dir not found: {mask_dir}")

    # 输出目录：<video>\mosaic_results\<mosaic_results-子目录名>
    suffix = img_dir.name
    out_dir = img_dir.parent / args.out_root / f"{args.out_root}-{suffix}-crop"
    ensure_dir(str(out_dir))

    bg_color = (255, 255, 255)
    color_rgb = tuple(args.color)
    valid_min_bytes = max(1, args.min_valid_kb) * 1024

    print("[INFO] IMG_DIR :", str(img_dir))
    print("[INFO] MASK_DIR:", str(mask_dir))
    print("[INFO] OUT_DIR :", str(out_dir))
    print(f"[INFO] Gap: row={args.gap_row}px, col={args.gap_col}px")
    print(f"[INFO] Overlay: darken={args.darken}, color={color_rgb}, alpha={args.alpha}")
    print(f"[INFO] ZoomCrop: zoom={args.zoom}, shift_x={args.shift_x}, shift_y={args.shift_y}")
    print(f"[INFO] Compress: <= {args.max_mb} MB, max_passes={args.max_passes}")
    print(f"[INFO] Resume: {args.resume}, Overwrite: {args.overwrite}, valid>= {args.min_valid_kb}KB")
    print("[INFO] Optimization: mask folder is NOT indexed; mask paths are checked by direct lookup.")

    map_img = index_images_only(str(img_dir))

    def has_all_img_frames(idx: str):
        return all(f in map_img.get(idx, {}) for f in range(N_FRAMES))

    candidate_ids = sorted([idx for idx in map_img.keys() if has_all_img_frames(idx)])

    if args.start_id is not None:
        s = str(args.start_id).zfill(6)
        candidate_ids = [i for i in candidate_ids if i >= s]
    if args.end_id is not None:
        e = str(args.end_id).zfill(6)
        candidate_ids = [i for i in candidate_ids if i <= e]
    if args.limit is not None:
        candidate_ids = candidate_ids[:args.limit]

    print(f"[INFO] Candidates with full 6 IMG frames: {len(candidate_ids)}")
    print("[INFO] Now checking masks on-the-fly per ID...")

    saved, skipped, skipped_exist, skipped_mask = 0, 0, 0, 0
    t_start = time.time()

    for idx in tqdm(candidate_ids):
        try:
            out_path = out_dir / f"{args.prefix}_{idx}.jpg"

            # resume
            if out_path.exists() and args.resume and (not args.overwrite):
                try:
                    if out_path.stat().st_size >= valid_min_bytes:
                        skipped_exist += 1
                        continue
                    else:
                        try:
                            out_path.unlink()
                        except:
                            pass
                except:
                    pass

            # check masks
            missing = []
            for f in range(N_FRAMES):
                mp = mask_path_for(str(mask_dir), idx, f)
                if not Path(mp).exists():
                    missing.append(f)
            if missing:
                skipped_mask += 1
                if args.verbose_missing:
                    print(f"\n[SKIP] {idx}: missing masks frames={missing}")
                continue

            # ===== 生成两列（先 zoom-crop，再 overlay）=====
            col_left = []
            col_right = []

            for f in range(N_FRAMES):
                # load image
                img = Image.open(map_img[idx][f]).convert("RGB")

                # load mask
                mp = mask_path_for(str(mask_dir), idx, f)
                mask = Image.open(mp).convert("L")
                if mask.size != img.size:
                    mask = mask.resize(img.size, resample=Image.NEAREST)

                # zoom-crop BOTH to keep alignment
                img_z = zoom_crop_keep_size(img, args.zoom, args.shift_x, args.shift_y, resample=Image.BICUBIC)
                mask_z = zoom_crop_keep_size(mask, args.zoom, args.shift_x, args.shift_y, resample=Image.NEAREST)

                col_left.append(img_z)
                col_right.append(overlay_mask_on_image(img_z, mask_z, args.darken, color_rgb, args.alpha))

            col_left_canvas = vstack(col_left, args.gap_row, bg_color)
            col_right_canvas = vstack(col_right, args.gap_row, bg_color)
            final_img = hstack([col_left_canvas, col_right_canvas], args.gap_col, bg_color)

            if out_path.exists() and args.overwrite:
                try:
                    out_path.unlink()
                except:
                    raise PermissionError(f"File locked: {out_path}")

            save_jpeg_under_limit(final_img, str(out_path), args.max_mb, args.max_passes)
            saved += 1

        except Exception as e:
            skipped += 1
            print(f"\n[SKIP] {idx}: {e}")
            time.sleep(0.02)
            continue

    dt = time.time() - t_start
    print("\nDone.")
    print("Saved:", saved)
    print("Skipped (errors):", skipped)
    print("Skipped (existing):", skipped_exist)
    print("Skipped (missing mask):", skipped_mask)
    print("Output:", str(out_dir))
    print(f"Time: {dt:.1f}s")


if __name__ == "__main__":
    main()
