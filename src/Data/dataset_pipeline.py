import os
import random
import librosa
import yt_dlp
import numpy as np
from pathlib import Path
from typing import List, Tuple

# ---------- 1) Download Audio --------------------
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



# ---------- 2) Run WhisperX ---------------------
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

    # return just sentence-level
    return aligned["segments"]   # <-- already have start/end/text


# ---------- 3) Select windows -------------------
def select_windows(duration, sr, y, window_sec=120):
    windows = []

    # ---- Window 1: first 2 min
    windows.append((0, min(window_sec, duration)))

    # ---- Window 2: mid windows
    mid_start = duration * 0.20
    mid_end   = duration * 0.80
    if mid_end - mid_start > window_sec:
        possible = np.arange(mid_start, mid_end-window_sec, window_sec)
        random_mid = random.sample(
            list(possible),
            k=min(3, len(possible))
        )
        for st in random_mid:
            windows.append((st, st+window_sec))

    # ---- Window 3: high-energy window
    frame_len = int(sr*0.05)
    hop_len   = int(sr*0.05)
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
    frames_per_win = int(window_sec / (hop_len/sr))

    for i in range(len(times)-frames_per_win):
        eng = rmse[i: i+frames_per_win].sum()
        if eng > best_energy:
            best_energy = eng
            best_start = times[i]
    windows.append((best_start, best_start + window_sec))

    # ---- Window 4: last 2 min
    windows.append((max(0, duration-window_sec), duration))

    return windows


# ---------- 4) Window sentence filter -----------
def get_sentences_in_window(segments, st, ed):
    out = []
    for seg in segments:
        if seg["start"] >= st and seg["end"] <= ed:
            out.append(seg)
    return out


# ---------- 5) Main selection & cap -------------
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

    # If > cap, random sample
    if len(selected) > max_cap:
        selected = random.sample(selected, max_cap)

    return selected


# ---------- MAIN PIPELINE -----------------------
def process_video(url, max_cap=300):
    audio_path, vid = download_audio(url)

    # load audio
    y, sr = librosa.load(audio_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)

    # whisperX
    segments = whisperx_transcribe(audio_path)

    # select windows
    windows = select_windows(duration, sr, y)

    # pick sentences
    final = select_sentences(segments, windows, max_cap=max_cap)

    return final, vid