# ========== GUIDE ==========
# Mode 1: Record new
#   python scripts/record_and_transcribe.py --speaker kyle --version real_v1
# Mode 2: Annotate existing manually
#   python scripts/record_and_transcribe.py --speaker kyle --version real_v1 --annotate-existing
# Mode 3: Auto rename and annotate
#   python scripts/record_and_transcribe.py --speaker kyle --version real_v1 --auto

import os
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import pandas as pd
import argparse
from datetime import datetime
import re
import torch
import torchaudio
import subprocess
from pathlib import Path

# === CONFIG ===
SAMPLE_RATE = 16000
CHANNELS = 1
DURATION = 5
BASE_DIR = "data/raw"
CSV_PATH = os.path.join(BASE_DIR, "real_dataset.csv")

# === HELPERS ===

def ensure_dirs(version_dir):
    audio_dir = os.path.join(BASE_DIR, "audio", version_dir)
    os.makedirs(audio_dir, exist_ok=True)
    return audio_dir

def load_dataset():
    return pd.read_csv(CSV_PATH) if os.path.exists(CSV_PATH) else pd.DataFrame(columns=["path", "text", "category", "speaker", "version", "timestamp"])

def load_transcriptions_from_txt(txt_path):
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    return []

def get_next_index(audio_dir, version, speaker):
    pattern = re.compile(rf"{version}_{speaker}_(\d{{4}})\.wav")
    indices = [int(m.group(1)) for f in os.listdir(audio_dir) if (m := pattern.match(f))]
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

def convert_to_model_ready(audio_path):
    output_path = audio_path.with_suffix(".wav")

    command = [
        "ffmpeg", "-y",
        "-i", str(audio_path),
        "-ar", str(SAMPLE_RATE),
        "-ac", "1",
        "-c:a", "pcm_s16le",
        str(output_path)
    ]

    print(f"[🔄] Converting {audio_path.name} to true WAV (16kHz mono)...")
    result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if result.returncode != 0:
        print(f"[❌] FFmpeg failed on: {audio_path}")
        return audio_path

    if audio_path.suffix.lower() != ".wav":
        os.remove(audio_path)

    return output_path

# === METADATA ===

def annotate(path, df, speaker, version, transcript=None):
    rel_path = os.path.relpath(path, BASE_DIR)
    if not transcript:
        transcript = input("Enter transcription (Bisaya): ").strip()
    else:
        print(f"[📜] Auto transcription: {transcript}")
    category = "phrase"
    timestamp = datetime.now().isoformat(timespec='seconds')
    df.loc[len(df)] = [rel_path, transcript, category, speaker, version, timestamp]
    df.to_csv(CSV_PATH, index=False)
    print(f"[📝] Metadata saved to {CSV_PATH}")
    return True

# === RECORD MODE ===

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

    return annotate(path, df, speaker, version)

# === ANNOTATION ===

def annotate_existing(audio_dir, df, speaker, version, transcriptions=None, auto=False):
    print(f"[📂] Annotating files in: {audio_dir}")
    audio_dir = Path(audio_dir)
    pattern = re.compile(rf"{version}_{speaker}_(\d{{4}})\.wav")
    next_index = get_next_index(audio_dir, version, speaker)
    transcription_index = 0

    for file in sorted(audio_dir.glob("*")):
        if not file.suffix.lower() in [".wav", ".m4a", ".mp3", ".ogg"]:
            continue

        # === Step 1: Convert first, always ===
        clean_file = convert_to_model_ready(file)
        if clean_file != file:
            file = clean_file

        # === Step 2: Rename only AFTER conversion ===
        match = pattern.match(file.name)
        if not match:
            new_name = generate_filename(version, speaker, next_index)
            print(f"[🔄] Renaming {file.name} → {new_name}")
            new_path = audio_dir / new_name
            file.rename(new_path)
            file = new_path
            next_index += 1

        # === Step 3: Skip if already annotated ===
        rel_path = os.path.relpath(file, BASE_DIR)
        if rel_path in df["path"].values:
            print(f"[⏩] Already annotated: {file.name}")
            continue

        print(f"\n[🎧] Playing: {file.name}")
        waveform, _ = torchaudio.load(str(file))
        playback(waveform[0].numpy())

        if auto:
            transcript = transcriptions[transcription_index] if transcription_index < len(transcriptions) else ""
            transcription_index += 1
            annotate(file, df, speaker, version, transcript)
        else:
            keep = input("Annotate this file? (y/n): ").strip().lower()
            if keep == 'y':
                annotate(file, df, speaker, version)

# === CLI ===

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record or Annotate Bisaya STT Audio")
    parser.add_argument("--version", type=str, default="real_v1", help="Dataset version tag (e.g., real_v1)")
    parser.add_argument("--speaker", type=str, required=True, help="Speaker ID (e.g., kyle, ana)")
    parser.add_argument("--annotate-existing", action="store_true", help="Annotate existing audio manually")
    parser.add_argument("--auto-rename", action="store_true", help="Auto rename audio files to match pattern")
    parser.add_argument("--auto-annotate", action="store_true", help="Auto-fill transcriptions from v1_clean.txt")
    parser.add_argument("--auto", action="store_true", help="Shortcut for --auto-rename and --auto-annotate")

    args = parser.parse_args()

    # Merge auto option into flags
    if args.auto:
        args.auto_rename = True
        args.auto_annotate = True

    df = load_dataset()
    audio_dir = ensure_dirs(args.version)
    txt_path = os.path.join(BASE_DIR, "v1_clean.txt")
    transcriptions = load_transcriptions_from_txt(txt_path) if args.auto_annotate else None

    if args.auto_rename or args.auto_annotate or args.annotate_existing:
        annotate_existing(
            audio_dir=audio_dir,
            df=df,
            speaker=args.speaker,
            version=args.version,
            transcriptions=transcriptions,
            auto=args.auto_annotate
        )
    else:
        index = get_next_index(audio_dir, args.version, args.speaker)
        while True:
            success = record_sample(index, args.version, args.speaker, df, audio_dir)
            if success:
                index += 1
            cont = input("Record another? (y/n): ").strip().lower()
            if cont != 'y':
                break
        print("[✅] All recordings completed.")
        print(f"[📂] Final dataset saved to {CSV_PATH}")
