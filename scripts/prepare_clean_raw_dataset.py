# scripts/prepare_clean_raw_dataset.py

import os
import pandas as pd
from datetime import datetime
from pathlib import Path

# === CONFIG ===
INPUT_CSV = "data/raw/bisaya-dataset/bisaya_dataset.csv"
OUTPUT_CSV = "data/raw/bisaya-dataset/bisaya_dataset_clean.csv"
RAW_AUDIO_DIR = "data/raw/audio"
VERSION = "real_v1"
SPEAKER = "kyle"
TIMESTAMP = datetime.now().isoformat(timespec="minutes")  # Example: 2025-07-22T16:41

# === Load CSV ===
df = pd.read_csv(INPUT_CSV)

# === Create versioned audio output dir ===
version_dir = Path(RAW_AUDIO_DIR) / VERSION
version_dir.mkdir(parents=True, exist_ok=True)

# === Process each entry ===
new_paths = []
missing = []

for i, row in df.iterrows():
    index = i + 1
    new_filename = f"{VERSION}_{SPEAKER}_{index:04}.wav"
    new_rel_path = f"audio/{VERSION}/{new_filename}"
    new_paths.append(new_rel_path)

    old_audio_path = Path(RAW_AUDIO_DIR) / f"sample{index}.wav"
    new_audio_path = version_dir / new_filename

    if old_audio_path.exists():
        os.rename(old_audio_path, new_audio_path)
        print(f"✅ {old_audio_path.name} → {new_filename}")
    else:
        print(f"⚠️ Missing: {old_audio_path.name}")
        missing.append(old_audio_path.name)

# === Add metadata ===
df["path"] = new_paths
df["speaker"] = SPEAKER
df["version"] = VERSION
df["timestamp"] = TIMESTAMP

# === Reorder columns ===
df = df[["path", "text", "category", "speaker", "version", "timestamp"]]

# === Save clean CSV ===
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False)

print(f"\n🎉 Done. Clean dataset saved to: {OUTPUT_CSV}")
if missing:
    print("⚠️ Some original audio files were missing:")
    for f in missing:
        print(f" - {f}")
