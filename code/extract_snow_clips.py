import os
import re
import math
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from openpyxl import Workbook


# =========================================================
# ✅ 你每次主要改这里（4个输入项）
# =========================================================

# (1) 读入：可以是单个 mp4，也可以是一个目录（会处理目录下所有 mp4）
INPUT_VIDEO_OR_DIR = r"XXXXXX"
# 例：单个视频
# INPUT_VIDEO_OR_DIR = r"XXXXXX"

# (2) 输出图片保存路径（建议单独放一个 images 目录）
OUTPUT_IMG_DIR = r"XXXXXX"

# (3) 起始 clip 编号（xxxxxx），可填 int 或 None
#     - 若为 None：自动从 OUTPUT_IMG_DIR 中已存在的最大编号继续
# START_CLIP_ID = None   # 例如从 120000 开始：START_CLIP_ID = 120000
START_CLIP_ID = 0

# (4) 裁剪：删掉前/后多少秒（默认 0）
TRIM_START_SEC = 3.0
TRIM_END_SEC   = 0.0

# 额外：如果 INPUT 是目录，你可以指定从某个视频文件名开始处理（可选）
# 例如从 "003.mp4" 开始：START_FROM_VIDEO_NAME = "003.mp4"
START_FROM_VIDEO_NAME = None

# =========================================================
# 固定协议：3s, 2fps -> 6帧
# =========================================================
CLIP_LEN_SEC = 3.0
FPS = 2
FRAMES_PER_CLIP = int(CLIP_LEN_SEC * FPS)  # 6

# log 目录（会自动新建）
LOG_DIR = r"XXXXXX


IMG_NAME_RE = re.compile(r"^img_(\d{6})_(\d)\.jpg$", re.IGNORECASE)


def run_cmd(cmd, check=True):
    """Run subprocess and capture output."""
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=False)
    if check and p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}):\n{' '.join(cmd)}\n\nSTDERR:\n{p.stderr}")
    return p.stdout, p.stderr, p.returncode


def ffprobe_info(video_path: str):
    """Return (duration_sec, size_bytes) via ffprobe + filesystem."""
    vp = Path(video_path)
    size_bytes = vp.stat().st_size

    # duration
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "format=duration",
        "-of", "json",
        str(vp)
    ]
    out, err, _ = run_cmd(cmd, check=True)
    data = json.loads(out)
    dur = float(data["format"]["duration"])
    return dur, size_bytes


def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def list_videos(input_path: str):
    p = Path(input_path)
    if p.is_file() and p.suffix.lower() in [".mp4", ".mov", ".mkv", ".webm", ".m4v"]:
        return [str(p)]
    if p.is_dir():
        vids = sorted([str(x) for x in p.glob("*.mp4")])
        return vids
    raise FileNotFoundError(f"INPUT_VIDEO_OR_DIR not found: {input_path}")


def auto_next_clip_id(output_dir: str):
    """Scan output_dir for existing img_XXXXXX_y.jpg and return next XXXXXX."""
    outp = Path(output_dir)
    if not outp.exists():
        return 0
    max_id = -1
    for f in outp.iterdir():
        if not f.is_file():
            continue
        m = IMG_NAME_RE.match(f.name)
        if m:
            cid = int(m.group(1))
            if cid > max_id:
                max_id = cid
    return max_id + 1


def make_unique_xlsx_name(log_dir: str, video_name: str):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = re.sub(r"[^\w\-.]+", "_", video_name)
    return str(Path(log_dir) / f"log_{ts}_{safe_name}.xlsx")


def extract_frames_for_video(
    video_path: str,
    output_dir: str,
    start_clip_id: int,
    trim_start: float,
    trim_end: float,
    clip_len: float,
    fps: int,
    frames_per_clip: int,
    log_dir: str
):
    """
    Strategy:
    1) 用 ffprobe 获取原始 duration
    2) 计算 effective_duration = duration - trim_start - trim_end
    3) 只取完整 clip：num_clips = floor(effective_duration / clip_len)
       used_duration = num_clips * clip_len
    4) ffmpeg 把 (trim_start ~ trim_start+used_duration) 这段按 fps=2 导出到 temp 连号图片
    5) Python 把 temp 序列按每 6 张分组重命名为 img_{xxxxxx}_{y}.jpg
    """
    ensure_dir(output_dir)
    ensure_dir(log_dir)

    video_name = Path(video_path).name
    t0 = datetime.now()

    duration, size_bytes = ffprobe_info(video_path)

    effective = max(0.0, duration - float(trim_start) - float(trim_end))
    num_clips = int(math.floor(effective / clip_len))
    used_duration = num_clips * clip_len
    total_frames = num_clips * frames_per_clip

    # 如果没有可用 clip，直接记录并返回
    status = "OK"
    note = ""
    if num_clips <= 0:
        status = "SKIP"
        note = f"effective_duration={effective:.3f}s too short for one clip ({clip_len}s)."
        # 写日志
        xlsx_path = make_unique_xlsx_name(log_dir, video_name)
        write_log_xlsx(
            xlsx_path=xlsx_path,
            rows=[{
                "processed_at": t0.strftime("%Y-%m-%d %H:%M:%S"),
                "video_path": video_path,
                "video_size_bytes": size_bytes,
                "duration_sec": duration,
                "trim_start_sec": trim_start,
                "trim_end_sec": trim_end,
                "effective_duration_sec": effective,
                "used_duration_sec": used_duration,
                "fps": fps,
                "clip_len_sec": clip_len,
                "frames_per_clip": frames_per_clip,
                "num_clips": num_clips,
                "total_frames": total_frames,
                "clip_id_start": start_clip_id,
                "clip_id_end": start_clip_id - 1,
                "output_dir": output_dir,
                "temp_dir": "",
                "ffmpeg_cmd": "",
                "status": status,
                "note": note,
            }]
        )
        return start_clip_id  # unchanged

    # temp dir
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_dir = str(Path(output_dir) / f"_tmp_{Path(video_name).stem}_{ts}")
    ensure_dir(temp_dir)

    # ffmpeg export
    # -ss trim_start
    # -t used_duration
    # -vf fps=2
    # 输出 tmp_%09d.jpg
    tmp_pattern = str(Path(temp_dir) / "tmp_%09d.jpg")

    ffmpeg_cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", str(trim_start),
        "-t", str(used_duration),
        "-i", video_path,
        "-vf", f"fps={fps}",
        "-q:v", "2",
        tmp_pattern
    ]

    try:
        _, _, _ = run_cmd(ffmpeg_cmd, check=True)
    except Exception as e:
        status = "FAIL"
        note = str(e)

    # 计数并重命名/移动
    clip_id_start = start_clip_id
    clip_id_end = start_clip_id + num_clips - 1

    if status == "OK":
        tmp_files = sorted(Path(temp_dir).glob("tmp_*.jpg"))
        if len(tmp_files) < total_frames:
            # 有时 ffmpeg 会因结尾取整少导出几张：我们按实际数量重新计算可用 clips
            actual_frames = len(tmp_files)
            actual_clips = actual_frames // frames_per_clip
            if actual_clips <= 0:
                status = "SKIP"
                note = f"ffmpeg produced {actual_frames} frames (<{frames_per_clip})."
                clip_id_end = start_clip_id - 1
                total_frames = actual_frames
                num_clips = actual_clips
            else:
                note = f"Warning: expected {total_frames} frames, got {actual_frames}. Use first {actual_clips} clips."
                num_clips = actual_clips
                total_frames = actual_clips * frames_per_clip
                used_duration = actual_clips * clip_len
                clip_id_end = start_clip_id + actual_clips - 1
                tmp_files = tmp_files[:total_frames]

        # 重命名输出
        if status == "OK":
            idx = 0
            for c in range(num_clips):
                cid = start_clip_id + c
                for y in range(frames_per_clip):
                    src = tmp_files[idx]
                    dst_name = f"img_{cid:06d}_{y}.jpg"
                    dst = Path(output_dir) / dst_name
                    # 避免覆盖
                    if dst.exists():
                        raise RuntimeError(f"Output file exists (would overwrite): {dst}")
                    shutil.move(str(src), str(dst))
                    idx += 1

    # 清理 temp（失败时保留现场也行，这里默认保留失败现场便于排查）
    if status == "OK":
        shutil.rmtree(temp_dir, ignore_errors=True)

    t1 = datetime.now()

    # 写日志 xlsx（每个视频一个）
    xlsx_path = make_unique_xlsx_name(log_dir, video_name)
    write_log_xlsx(
        xlsx_path=xlsx_path,
        rows=[{
            "processed_at": t0.strftime("%Y-%m-%d %H:%M:%S"),
            "finished_at": t1.strftime("%Y-%m-%d %H:%M:%S"),
            "video_path": video_path,
            "video_name": video_name,
            "video_size_bytes": size_bytes,
            "duration_sec": duration,
            "trim_start_sec": trim_start,
            "trim_end_sec": trim_end,
            "effective_duration_sec": effective,
            "used_duration_sec": used_duration,
            "fps": fps,
            "clip_len_sec": clip_len,
            "frames_per_clip": frames_per_clip,
            "num_clips": num_clips,
            "total_frames": total_frames,
            "clip_id_start": clip_id_start,
            "clip_id_end": clip_id_end,
            "output_dir": output_dir,
            "temp_dir": temp_dir if status != "OK" else "",
            "ffmpeg_cmd": " ".join(ffmpeg_cmd),
            "status": status,
            "note": note,
        }]
    )

    # 返回下一段起始编号（续接用）
    if status == "OK":
        return clip_id_end + 1
    else:
        return start_clip_id  # 失败不前进，避免编号断层（你也可以改成失败也前进）


def write_log_xlsx(xlsx_path: str, rows: list):
    wb = Workbook()
    ws = wb.active
    ws.title = "log"

    # header
    headers = list(rows[0].keys())
    ws.append(headers)
    for r in rows:
        ws.append([r.get(h, "") for h in headers])

    # basic formatting: freeze first row
    ws.freeze_panes = "A2"

    wb.save(xlsx_path)


def main():
    ensure_dir(LOG_DIR)
    ensure_dir(OUTPUT_IMG_DIR)

    videos = list_videos(INPUT_VIDEO_OR_DIR)
    if not videos:
        print("No videos found.")
        return

    # 如果指定从某个视频名开始
    if START_FROM_VIDEO_NAME is not None:
        start_name = START_FROM_VIDEO_NAME.lower()
        found = False
        new_list = []
        for v in videos:
            if Path(v).name.lower() == start_name:
                found = True
            if found:
                new_list.append(v)
        if not new_list:
            raise RuntimeError(f"START_FROM_VIDEO_NAME not found in directory: {START_FROM_VIDEO_NAME}")
        videos = new_list

    # 起始编号
    if START_CLIP_ID is None:
        start_clip_id = auto_next_clip_id(OUTPUT_IMG_DIR)
    else:
        start_clip_id = int(START_CLIP_ID)

    print(f"[INFO] Found {len(videos)} video(s).")
    print(f"[INFO] Output dir: {OUTPUT_IMG_DIR}")
    print(f"[INFO] Start clip id: {start_clip_id:06d}")
    print(f"[INFO] Trim: start={TRIM_START_SEC}s, end={TRIM_END_SEC}s")
    print(f"[INFO] Protocol: {CLIP_LEN_SEC}s @ {FPS}fps => {FRAMES_PER_CLIP} frames/clip")

    for i, v in enumerate(videos, 1):
        print(f"\n=== ({i}/{len(videos)}) Processing: {v}")
        next_id = extract_frames_for_video(
            video_path=v,
            output_dir=OUTPUT_IMG_DIR,
            start_clip_id=start_clip_id,
            trim_start=TRIM_START_SEC,
            trim_end=TRIM_END_SEC,
            clip_len=CLIP_LEN_SEC,
            fps=FPS,
            frames_per_clip=FRAMES_PER_CLIP,
            log_dir=LOG_DIR
        )
        print(f"[INFO] Done. Next clip id would be: {next_id:06d}")
        start_clip_id = next_id

    print("\nAll done.")


if __name__ == "__main__":
    main()
