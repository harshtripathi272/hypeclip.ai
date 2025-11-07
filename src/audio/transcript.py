# import whisperx
# import torch
# import os


# VIDEO_PATH = "The Napkin Ring Problem.mp4"   # your input video
# LANGUAGE = "en"                   # specify manually or leave None for auto-detect
# MODEL_SIZE = "large-v2"           # can be tiny / base / small / medium / large-v2





# device = "cpu"
# model = whisperx.load_model("large-v2", device=device, compute_type="int8")  # ✅ works on CPU


# # --- STEP 2: Transcribe audio from video ---
# print("Transcribing...")
# audio = whisperx.load_audio(VIDEO_PATH)
# result = model.transcribe(audio, batch_size=16, language=LANGUAGE)

# # --- STEP 3: Load alignment model ---
# print("Loading alignment model...")
# model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)

# # --- STEP 4: Align whisper output at word level ---
# print("Aligning words...")
# result_aligned = whisperx.align(
#     result["segments"],
#     model_a,
#     metadata,
#     audio,
#     device,
#     return_char_alignments=False
# )

# # --- STEP 5: Print word-level timestamps ---
# print("\nWord-level timestamps:")
# for segment in result_aligned["segments"]:
#     for word in segment["words"]:
#         print(f"[{word['start']:.2f}s - {word['end']:.2f}s] {word['word']}")

# # --- Optional: Save results ---
# os.makedirs("output", exist_ok=True)
# import json
# with open("output/transcript_word_level.json", "w", encoding="utf-8") as f:
#     json.dump(result_aligned, f, ensure_ascii=False, indent=2)

# print("\n✅ Word-level transcript saved to output/transcript_word_level.json")
import whisperx
import torch
import os
import json

# --- CONFIG ---
VIDEO_PATH = "Videos/World's Most Expensive Flights! [1WEAJ-DFkHE].mp4"  # input video
LANGUAGE = "en"                             # manually specify or set to None for auto-detect
MODEL_SIZE = "large-v2"                     # tiny / base / small / medium / large-v2

# --- STEP 1: Device Setup ---
if torch.cuda.is_available():
    device = "cuda"
    compute_type = "float16"  # ✅ Recommended for GPU (fast + efficient)
    print("🔥 Using GPU for WhisperX")
else:
    device = "cpu"
    compute_type = "int8"     # fallback for CPU
    print("⚙️ Using CPU")

# --- STEP 2: Load Whisper Model ---
print("Loading WhisperX model...")
model = whisperx.load_model(MODEL_SIZE, device=device, compute_type=compute_type)

# --- STEP 3: Transcribe audio ---
print("Transcribing...")
audio = whisperx.load_audio(VIDEO_PATH)
result = model.transcribe(audio, batch_size=16, language=LANGUAGE)

# --- STEP 4: Load Alignment Model ---
print("Loading alignment model...")
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)

# --- STEP 5: Align word-level timestamps ---
print("Aligning words...")
result_aligned = whisperx.align(
    result["segments"],
    model_a,
    metadata,
    audio,
    device,
    return_char_alignments=False
)

# --- STEP 6: Print word-level timestamps ---
print("\nWord-level timestamps:")
for segment in result_aligned["segments"]:
    for word in segment["words"]:
        print(f"[{word['start']:.2f}s - {word['end']:.2f}s] {word['word']}")

# --- STEP 7: Save Results ---
os.makedirs("output", exist_ok=True)
with open("output/transcript_word_level_Mr_Beast.json", "w", encoding="utf-8") as f:
    json.dump(result_aligned, f, ensure_ascii=False, indent=2)

print("\n✅ Word-level transcript saved to output/transcript_word_level.json")
