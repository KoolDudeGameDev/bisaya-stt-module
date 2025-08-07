import pandas as pd
import torchaudio
from pathlib import Path
import re

# Load the dataset CSV
df = pd.read_csv("data/raw/real_dataset.csv")

# Define tagging function based on audio duration
def tag_length_by_duration(path):
    try:
        waveform, sample_rate = torchaudio.load(path)
        duration_sec = waveform.size(1) / sample_rate
        return "short" if duration_sec <= 2.5 else "long"
    except Exception as e:
        print(f"Failed to process {path}: {e}")
        return "unknown"

# Apply duration-based tagging
df["length"] = df["path"].apply(tag_length_by_duration)

# Clean text column (optional, consistent with past logic)
def clean_text(text):
    return re.sub(r"[,.?!;:]", "", text)

df["text"] = df["text"].apply(clean_text)

# Save updated CSV
output_path = "data/raw/real_dataset_tagged.csv"
df.to_csv(output_path, index=False)

print(f"Updated CSV saved to: {output_path}")
