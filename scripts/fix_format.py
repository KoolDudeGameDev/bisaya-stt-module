# scripts/fix_format.py

import pandas as pd
from datetime import datetime
import os

# === Config ===
INPUT_CSV = "data/raw/bisaya-dataset/bisaya_dataset.csv"
OUTPUT_CSV = "data/raw/bisaya-dataset/bisaya_dataset_clean.csv"
SPEAKER = "kyle"
VERSION = "real_v1"
TIMESTAMP = datetime.now().isoformat(timespec='minutes')  # e.g., "2025-07-22T16:41"

# === Load the Dataset ===
df = pd.read_csv(INPUT_CSV)

# === Update Columns ===
# Fix the path to include version prefix and folder
def format_path(index):
    return f"audio/{VERSION}/{VERSION}_{SPEAKER}_{index+1:04}.wav"

df["path"] = [format_path(i) for i in range(len(df))]
df["speaker"] = SPEAKER
df["version"] = VERSION
df["timestamp"] = TIMESTAMP

# === Reorder Columns ===
df = df[["path", "text", "category", "speaker", "version", "timestamp"]]

# === Output Clean CSV ===
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False)

print(f"✅ Cleaned dataset saved to: {OUTPUT_CSV}")
