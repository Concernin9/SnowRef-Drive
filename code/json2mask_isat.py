import os
import re
import json
import argparse
from pathlib import Path

from PIL import Image, ImageDraw
from tqdm import tqdm


IMG_RE = re.compile(r"^img_(\d{6})_(\d)\.jpg$", re.IGNORECASE)

# 认为“有效输出 mask 文件”的最小大小（避免 0KB 坏文件被跳过）
VALID_OUT_MIN_BYTES = 200  # PNG 很小，200B 足够


def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def find_images(img_dir: Path):
    """
    扫描 img_dir 下的 img_XXXXXX_y.jpg
    返回：[(id_str, frame_idx, img_path), ...]
    """
    items = []
    for p in img_dir.iterdir():
        if not p.is_file():
            continue
        m = IMG_RE.match(p.name)
        if not m:
            continue
        idx = m.group(1)
        fr = int(m.group(2))
        items.append((idx, fr, p))
    items.sort()
    return items


def load_json_safe(json_path: Path):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def polygon_to_mask(width: int, height: int, polygons):
    """
    polygons: list of list of (x,y)
    输出 L 模式 mask：背景0，前景255
    """
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    for poly in polygons:
        if not poly or len(poly) < 3:
            continue
        pts = [(float(x), float(y)) for x, y in poly]
        draw.polygon(pts, fill=255)

    return mask


def main():
    ap = argparse.ArgumentParser(description="Convert ISAT json polygons to binary PNG masks (always output black if no json).")

    ap.add_argument(
        "--in_dir",
        default=r"XXXXXX",
        help="Input folder containing img_XXXXXX_y.jpg and (optional) img_XXXXXX_y.json"
    )
    ap.add_argument(
        "--out_dir",
        default=r"XXXXXX",
        help="Output folder for mask_XXXXXX_y.png"
    )

    ap.add_argument("--skip_existing", action="store_true", default=True,
                    help="Skip if output mask exists and looks valid (default: on)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Force overwrite existing masks")
    ap.add_argument("--min_valid_bytes", type=int, default=VALID_OUT_MIN_BYTES,
                    help="If skip_existing is on, output file must be >= this size to be considered valid")

    ap.add_argument("--verbose", action="store_true",
                    help="Print details for missing/empty json")

    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)

    if not in_dir.exists():
        raise FileNotFoundError(f"in_dir not found: {in_dir}")
    ensure_dir(str(out_dir))

    items = find_images(in_dir)
    print(f"[INFO] Found images: {len(items)}")
    print(f"[INFO] in_dir : {in_dir}")
    print(f"[INFO] out_dir: {out_dir}")
    print(f"[INFO] skip_existing={args.skip_existing}, overwrite={args.overwrite}, min_valid_bytes={args.min_valid_bytes}")

    saved = 0
    skipped_exist = 0
    black_missing_json = 0
    black_empty_poly = 0
    ok_poly = 0
    errors = 0

    for idx, fr, img_path in tqdm(items):
        try:
            out_path = out_dir / f"mask_{idx}_{fr}.png"

            # ===== 断点续跑：存在且正常大小 -> 跳过 =====
            if out_path.exists() and args.skip_existing and (not args.overwrite):
                try:
                    if out_path.stat().st_size >= args.min_valid_bytes:
                        skipped_exist += 1
                        continue
                    else:
                        # 太小认为坏文件：删掉重算
                        try:
                            out_path.unlink()
                        except:
                            pass
                except:
                    pass

            # 读图尺寸（最稳）
            with Image.open(img_path) as im:
                w, h = im.size

            # 预设：默认黑 mask（无论有没有 json，最后都要输出）
            mask = Image.new("L", (w, h), 0)

            json_path = img_path.with_suffix(".json")
            if not json_path.exists():
                # 没 json：直接输出黑 mask
                black_missing_json += 1
                if args.verbose:
                    print(f"\n[NO JSON -> BLACK] {json_path.name}")
                mask.save(out_path)
                saved += 1
                continue

            data = load_json_safe(json_path)
            if data is None:
                # json 读失败：也输出黑
                black_missing_json += 1
                if args.verbose:
                    print(f"\n[BAD JSON -> BLACK] {json_path.name}")
                mask.save(out_path)
                saved += 1
                continue

            objs = data.get("objects", [])
            polygons = []
            for obj in objs:
                seg = obj.get("segmentation", None)
                if seg and len(seg) >= 3:
                    polygons.append(seg)

            if not polygons:
                # 有 json 但没 polygon：输出黑
                black_empty_poly += 1
                if args.verbose:
                    print(f"\n[EMPTY POLY -> BLACK] {json_path.name}")
                mask.save(out_path)
                saved += 1
                continue

            # 有 polygon：画出来
            mask = polygon_to_mask(w, h, polygons)
            mask.save(out_path)
            saved += 1
            ok_poly += 1

        except Exception as e:
            errors += 1
            if args.verbose:
                print(f"\n[ERROR] {idx}_{fr}: {e}")
            continue

    print("\nDone.")
    print("Saved:", saved)
    print("Skipped existing:", skipped_exist)
    print("Poly masks:", ok_poly)
    print("Black (missing/bad json):", black_missing_json)
    print("Black (empty polygons):", black_empty_poly)
    print("Errors:", errors)
    print("Output:", str(out_dir))


if __name__ == "__main__":
    main()
