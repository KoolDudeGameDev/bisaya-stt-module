# ========== GUIDE ==========
# python scripts/record_and_transcribe.py --speaker kyle --version real_v1

import os
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import pandas as pd
import argparse
from datetime import datetime
import re

# === CONFIG ===
SAMPLE_RATE = 16000
CHANNELS = 1
DURATION = 5  # seconds
BASE_DIR = "bisaya-dataset"
CSV_PATH = os.path.join(BASE_DIR, "bisaya_dataset.csv")

# === HELPER FUNCTIONS ===

def ensure_dirs(version_dir):
    audio_dir = os.path.join(BASE_DIR, "audio", version_dir)
    os.makedirs(audio_dir, exist_ok=True)
    return audio_dir

def load_dataset():
    if os.path.exists(CSV_PATH):
        return pd.read_csv(CSV_PATH)
    return pd.DataFrame(columns=["path", "text", "category", "speaker", "version", "timestamp"])

def get_next_index(audio_dir, version, speaker):
    """Scans directory for highest existing index for a version/speaker"""
    if not os.path.exists(audio_dir):
        return 1

    pattern = re.compile(rf"{version}_{speaker}_(\d{{4}})\.wav")
    indices = []

    for filename in os.listdir(audio_dir):
        match = pattern.match(filename)
        if match:
            indices.append(int(match.group(1)))

    return max(indices, default=0) + 1

def generate_filename(version, speaker, index):
    return f"{version}_{speaker}_{str(index).zfill(4)}.wav"

def record_audio(duration):
    print(f"\n[🎙️] Recording for {duration} seconds...")
    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16')
    sd.wait()
    return audio

def playback(audio):
    print("[🔊] Playing back the recording...")
    sd.play(audio, SAMPLE_RATE)
    sd.wait()

# === MAIN FUNCTION ===

def record_sample(index, version, speaker, df, audio_dir):
    filename = generate_filename(version, speaker, index)
    path = os.path.join(audio_dir, filename)

    if os.path.exists(path):
        print(f"[⚠️] File already exists: {path}. Skipping.")
        return False

    audio = record_audio(DURATION)
    wav.write(path, SAMPLE_RATE, audio)
    print(f"[✅] Saved to {path}")

    playback(audio)
    review = input("Keep this recording? (y/n): ").strip().lower()
    if review != 'y':
        os.remove(path)
        print("[🗑️] Recording deleted.")
        return False

    # Metadata input
    transcript = input("Enter transcription (Bisaya): ").strip()
    category = input("Enter category (e.g., number, command): ").strip().lower()

    rel_path = os.path.relpath(path, BASE_DIR)
    timestamp = datetime.now().isoformat(timespec='seconds')
    df.loc[len(df.index)] = [rel_path, transcript, category, speaker, version, timestamp]
    df.to_csv(CSV_PATH, index=False)
    print(f"[📝] Entry saved to {CSV_PATH}")
    return True

# === CLI ===

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record Bisaya STT Samples")
    parser.add_argument("--version", type=str, default="real_v1", help="Dataset version tag (e.g., real_v1)")
    parser.add_argument("--speaker", type=str, required=True, help="Speaker ID (e.g., kyle, ana)")
    args = parser.parse_args()

    df = load_dataset()
    audio_dir = ensure_dirs(args.version)
    start_index = get_next_index(audio_dir, args.version, args.speaker)

    while True:
        success = record_sample(start_index, args.version, args.speaker, df, audio_dir)
        if success:
            start_index += 1
        cont = input("Record another? (y/n): ").strip().lower()
        if cont != 'y':
            break
