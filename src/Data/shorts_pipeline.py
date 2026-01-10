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

    # Attempt to extract ID quickly to check if file exists
    vid_id = None
    try:
        if "v=" in url:
            vid_id = url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in url:
            vid_id = url.split("youtu.be/")[1].split("?")[0]
        elif "shorts/" in url:
            vid_id = url.split("shorts/")[1].split("?")[0]
    except:
        pass

    # If we guessed the ID and the file exists, SKIP download!
    if vid_id:
        audio_path = f"{out_dir}/{vid_id}.wav"
        if os.path.exists(audio_path):
            print(f"🎬 Found existing audio: {audio_path} (Skipping download)")
            return audio_path, vid_id

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


def select_windows_shorts(duration):
    """
    For YouTube Shorts (usually < 60s), we just take the entire video duration.
    No complex window selection is needed.
    """
    return [(0, duration)]


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


def process_video_shorts(url, max_cap=300):
    audio_path, vid = download_audio(url)

    y, sr = librosa.load(audio_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)

    segments = whisperx_transcribe(audio_path)

    # Simplified window selection for Shorts
    windows = select_windows_shorts(duration)

    final = select_sentences(segments, windows, max_cap=max_cap)

    return final, vid



# ================= MAIN EXECUTION FOR SHORTS =================

# Use short2.txt as input
urls = read_urls("short2.txt")


# Load existing progress to support resuming
processed_ids = set()
# Separate output file for shorts
output_file_jsonl = "output/shorts_dataset.jsonl"
os.makedirs("output", exist_ok=True)

if os.path.exists(output_file_jsonl):
    with open(output_file_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                processed_ids.add(data.get("video_id"))
            except:
                continue

# Open in append mode ("a") to save incrementally
with open(output_file_jsonl, "a", encoding="utf-8") as f_jsonl:
    for url in urls:
        # Optimization: Extract ID from URL directly to avoid network call
        try:
            if "v=" in url:
                vid_id = url.split("v=")[1].split("&")[0]
            elif "youtu.be/" in url:
                vid_id = url.split("youtu.be/")[1].split("?")[0]
            elif "shorts/" in url:
                vid_id = url.split("shorts/")[1].split("?")[0]
            else:
                # Fallback to yt-dlp if URL format is complex
                with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                    info = ydl.extract_info(url, download=False)
                    vid_id = info['id']
            
            if vid_id in processed_ids:
                print(f"⏩ Skipping {vid_id} (Already processed)")
                continue

            segments, vid_id = process_video_shorts(url)
            
            for seg in segments:
                seg["video_id"] = vid_id
                # Save each segment to JSONL immediately
                f_jsonl.write(json.dumps(seg, ensure_ascii=False) + "\n")
            
            f_jsonl.flush() # Ensure it's written to disk
            print(f"✅ Processed and SAVED video: {vid_id}, segments: {len(segments)}")
            
        except Exception as e:
            print(f"❌ Failed to process {url}: {e}")

# After the loop, generate the other descriptive formats from the JSONL file
print("📊 Generating CSV, JSON, and TXT summaries for Shorts...")
all_data = []
if os.path.exists(output_file_jsonl):
    with open(output_file_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            all_data.append(json.loads(line))


import csv

output_file_json = "output/shorts_dataset.json"
with open(output_file_json, "w", encoding="utf-8") as f:
    json.dump(all_data, f, ensure_ascii=False, indent=2)

output_file_csv = "output/shorts_dataset.csv"
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

output_file_txt = "output/shorts_dataset.txt"
with open(output_file_txt, "w", encoding="utf-8") as f:
    for seg in all_data:
        f.write(seg.get("text", "") + "\n")

print(f"✅ Shorts Dataset saved in 'output/shorts_dataset.*' formats")
