import os
import torch
import omegaconf

torch.serialization.add_safe_globals([
    omegaconf.listconfig.ListConfig,
    omegaconf.dictconfig.DictConfig,
    omegaconf.base.ContainerMetadata,
])


import random
import librosa
import yt_dlp
import numpy as np
import json
from pathlib import Path
from typing import List, Tuple


def read_urls(file_path: str):
    with open(file_path, "r") as f:
        urls = [line.strip() for line in f if line.strip()]
    return urls

def download_audio(url: str, out_dir="downloads"):
    os.makedirs(out_dir, exist_ok=True)

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl': f'{out_dir}/%(id)s.%(ext)s'
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

    vid = info["id"]
    audio_path = f"{out_dir}/{vid}.wav"
    return audio_path, vid


def whisperx_transcribe(audio_path, device="cuda"):
    import whisperx
    model = whisperx.load_model("large-v2", device)
    result = model.transcribe(audio_path)

    # Align
    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"],
        device=device
    )
    aligned = whisperx.align(
        result["segments"], model_a, metadata,
        audio_path, device
    )

    return aligned["segments"]


def select_windows(duration, sr, y, window_sec=120):
    windows = []


    windows.append((0, min(window_sec, duration)))


    mid_start = duration * 0.20
    mid_end = duration * 0.80
    if mid_end - mid_start > window_sec:
        possible = np.arange(mid_start, mid_end-window_sec, window_sec)
        random_mid = random.sample(
            list(possible),
            k=min(3, len(possible))
        )
        for st in random_mid:
            windows.append((st, st+window_sec))

    frame_len = int(sr * 0.05)
    hop_len = int(sr * 0.05)
    rmse = librosa.feature.rms(
        y=y,
        frame_length=frame_len,
        hop_length=hop_len
    )[0]
    times = librosa.frames_to_time(
        np.arange(len(rmse)),
        sr=sr,
        hop_length=hop_len
    )
    best_energy = -1
    best_start = 0
    frames_per_win = int(window_sec / (hop_len / sr))

    for i in range(len(times) - frames_per_win):
        eng = rmse[i: i + frames_per_win].sum()
        if eng > best_energy:
            best_energy = eng
            best_start = times[i]
    windows.append((best_start, best_start + window_sec))

    windows.append((max(0, duration - window_sec), duration))

    return windows



def get_sentences_in_window(segments, st, ed):
    out = []
    for seg in segments:
        if seg["start"] >= st and seg["end"] <= ed:
            out.append(seg)
    return out


def select_sentences(segments, windows, max_cap=300):
    selected = []
    seen = set()

    for (st, ed) in windows:
        chunk = get_sentences_in_window(segments, st, ed)
        for c in chunk:
            key = (c["start"], c["end"])
            if key not in seen:
                seen.add(key)
                selected.append(c)

    if len(selected) > max_cap:
        selected = random.sample(selected, max_cap)

    return selected


def process_video(url, max_cap=300):
    audio_path, vid = download_audio(url)

    y, sr = librosa.load(audio_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)

    segments = whisperx_transcribe(audio_path)

    windows = select_windows(duration, sr, y)

    final = select_sentences(segments, windows, max_cap=max_cap)

    return final, vid



urls = read_urls("url.txt")


all_data = []


for url in urls:
    try:
        segments, vid_id = process_video(url)
        
 
        for seg in segments:
            seg["video_id"] = vid_id
        
        
        all_data.extend(segments)
        print(f"✅ Processed video: {vid_id}, segments: {len(segments)}")
    
    except Exception as e:
        print(f"❌ Failed to process {url}: {e}")


import csv


output_file_jsonl = "output/dataset_long_1.jsonl"
with open(output_file_jsonl, "w", encoding="utf-8") as f:
    for seg in all_data:
        f.write(json.dumps(seg, ensure_ascii=False) + "\n")


output_file_json = "output/dataset_long_1.json"
with open(output_file_json, "w", encoding="utf-8") as f:
    json.dump(all_data, f, ensure_ascii=False, indent=2)

output_file_csv = "output/dataset_long_1.csv"
with open(output_file_csv, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["video_id", "start", "end", "text"])
    writer.writeheader()
    for seg in all_data:
        writer.writerow({
            "video_id": seg.get("video_id", ""),
            "start": seg.get("start", ""),
            "end": seg.get("end", ""),
            "text": seg.get("text", "")
        })

output_file_txt = "output/dataset_long_1.txt"
with open(output_file_txt, "w", encoding="utf-8") as f:
    for seg in all_data:
        f.write(seg.get("text", "") + "\n")

print(f"✅ Dataset saved in JSONL, JSON, CSV, and TXT formats")
