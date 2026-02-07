import re
import argparse
from pathlib import Path

PAT = re.compile(r"^mask_(\d{6})_(\d)\.png$", re.IGNORECASE)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mask_dir",
        type=str,
        default=r"XXXXXX",
        help="Directory containing mask_XXXXXX_Y.png",
    )
    parser.add_argument(
        "--n_frames",
        type=int,
        default=6,
        help="How many frames per clip (default 6 => y in 0..5)",
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        default="missing_report",
        help="Output txt prefix (will create multiple txt files)",
    )
    args = parser.parse_args()

    mask_dir = Path(args.mask_dir)
    if not mask_dir.exists():
        raise FileNotFoundError(f"mask_dir not found: {mask_dir}")

    # id -> set(frame_idx)
    id2frames = {}

    total_files = 0
    bad_names = 0

    for p in mask_dir.iterdir():
        if not p.is_file():
            continue
        total_files += 1
        m = PAT.match(p.name)
        if not m:
            bad_names += 1
            continue
        clip_id = int(m.group(1))
        frame_id = int(m.group(2))
        id2frames.setdefault(clip_id, set()).add(frame_id)

    if not id2frames:
        raise RuntimeError("No valid mask_XXXXXX_Y.png files found. Check directory and naming rule.")

    min_id = min(id2frames.keys())
    max_id = max(id2frames.keys())

    expected_frames = set(range(args.n_frames))

    # 1) missing clip ids in [min_id, max_id]
    missing_ids = [i for i in range(min_id, max_id + 1) if i not in id2frames]

    # 2) incomplete clips (exists but missing some frames)
    incomplete = []
    for clip_id, frames in sorted(id2frames.items()):
        miss = sorted(expected_frames - frames)
        extra = sorted(frames - expected_frames)
        if miss or extra:
            incomplete.append((clip_id, miss, extra))

    # write outputs
    out_dir = mask_dir.parent  # save next to [mask]-all-now (i.e., ...\mask\)
    out1 = out_dir / f"{args.out_prefix}_missing_clip_ids_{min_id:06d}-{max_id:06d}.txt"
    out2 = out_dir / f"{args.out_prefix}_incomplete_clips.txt"
    out3 = out_dir / f"{args.out_prefix}_summary.txt"

    with out1.open("w", encoding="utf-8") as f:
        for i in missing_ids:
            f.write(f"{i:06d}\n")

    with out2.open("w", encoding="utf-8") as f:
        for clip_id, miss, extra in incomplete:
            f.write(f"{clip_id:06d}  missing={miss}  extra={extra}\n")

    with out3.open("w", encoding="utf-8") as f:
        f.write(f"mask_dir: {mask_dir}\n")
        f.write(f"total_files_scanned: {total_files}\n")
        f.write(f"bad_name_files: {bad_names}\n")
        f.write(f"valid_clips: {len(id2frames)}\n")
        f.write(f"id_range: {min_id:06d} ~ {max_id:06d}\n")
        f.write(f"missing_clip_ids_in_range: {len(missing_ids)}\n")
        f.write(f"incomplete_clips: {len(incomplete)}\n")

    print("Done.")
    print(f"Range: {min_id:06d} ~ {max_id:06d}")
    print(f"Missing clip ids: {len(missing_ids)} -> {out1}")
    print(f"Incomplete clips : {len(incomplete)} -> {out2}")
    print(f"Summary          : {out3}")
    if bad_names:
        print(f"Warning: {bad_names} files did not match pattern mask_XXXXXX_Y.png")

if __name__ == "__main__":
    main()
