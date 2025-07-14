# scripts/review_manifest.py

import os
import re
import pandas as pd
import librosa
from tqdm import tqdm

def load_manifest(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Manifest not found: {path}")
    df = pd.read_csv(path)
    print(f"[INFO] Loaded manifest: {len(df)} entries")
    return df

def check_nulls(df):
    nulls = df.isnull().sum()
    print("[CHECK] Missing fields:")
    print(nulls[nulls > 0])

def check_missing_audio(df):
    missing = [p for p in df['path'] if not os.path.exists(p)]
    print(f"[CHECK] Missing audio files: {len(missing)}")
    if missing:
        for m in missing[:10]:
            print(f"  - {m}")
    return missing

def check_invalid_text(df):
    pattern = re.compile(r"[^a-zA-Z\sñÑ]")
    invalid = df[df['text'].apply(lambda x: bool(pattern.search(str(x))))]
    print(f"[CHECK] Invalid text entries: {len(invalid)}")
    return invalid

def check_audio_durations(df, sample_size=50):
    durations = []
    subset = df.sample(min(sample_size, len(df)), random_state=42)
    for _, row in tqdm(subset.iterrows(), total=len(subset)):
        try:
            y, sr = librosa.load(row['path'], sr=None)
            durations.append(len(y) / sr)
        except Exception as e:
            print(f"❌ Error loading {row['path']}: {e}")
    if durations:
        print(f"[CHECK] Avg duration: {sum(durations)/len(durations):.2f}s")
        print(f"         Min: {min(durations):.2f}s | Max: {max(durations):.2f}s")

def save_clean_manifest(df, invalid_rows, missing_files, output_path):
    clean_df = df[~df.index.isin(invalid_rows.index)]
    clean_df = clean_df[~clean_df['path'].isin(missing_files)]
    clean_df.to_csv(output_path, index=False)
    print(f"[SAVE] Cleaned manifest saved to: {output_path}")

def main():
    manifest_path = "data/synthetic/manifests/manifest_synthetic_cebuano_v1.csv"
    output_path = "data/synthetic/manifests/manifest_synthetic_cleaned.csv"

    df = load_manifest(manifest_path)
    check_nulls(df)

    missing_files = check_missing_audio(df)
    invalid_text = check_invalid_text(df)
    check_audio_durations(df)

    if missing_files or len(invalid_text):
        save_clean_manifest(df, invalid_text, missing_files, output_path)
    else:
        print("[✅] Manifest is clean. No need to save cleaned version.")

if __name__ == "__main__":
    main()
