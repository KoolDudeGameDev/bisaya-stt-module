import os
import pandas as pd
import soundfile as sf

# ========== CONFIG ==========
CSV_PATH = "data/raw/bisaya-dataset/bisaya_dataset.csv"
AUDIO_DIR = "data/raw/audio"
CLEAN_CSV_OUTPUT = "data/raw/bisaya-dataset/bisaya_dataset_clean.csv"
# ============================


def get_duration_sec(wav_path):
    try:
        info = sf.info(wav_path)
        return round(info.duration, 3)
    except Exception as e:
        print(f"[❌] Error reading duration for {wav_path}: {e}")
        return None


def validate_and_augment_dataset():
    if not os.path.exists(CSV_PATH):
        print(f"[❌] CSV not found at: {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)

    required_columns = {"path", "text"}
    if not required_columns.issubset(df.columns):
        print(f"[❌] CSV must contain at least 'path' and 'text' columns.")
        return

    print(f"\n[📊] Total entries in CSV: {len(df)}")

    seen_paths = set()
    missing_files = []
    duplicates = []
    bad_rows = []
    durations = []

    for idx, row in df.iterrows():
        relative_path = row["path"]
        full_path = os.path.join(AUDIO_DIR, os.path.basename(relative_path))
        text = str(row["text"]).strip()

        # Check duplicates
        if relative_path in seen_paths:
            duplicates.append(idx)
            continue
        else:
            seen_paths.add(relative_path)

        # Check file existence
        if not os.path.exists(full_path):
            missing_files.append(idx)
            continue

        # Check empty fields
        if not text:
            bad_rows.append(idx)
            continue

        # Get duration
        duration = get_duration_sec(full_path)
        if duration is None:
            bad_rows.append(idx)
            continue

        durations.append(duration)

    total_invalid = set(missing_files + duplicates + bad_rows)
    print(f"[✅] Valid entries: {len(df) - len(total_invalid)}")
    print(f"[⚠️] Missing files: {len(missing_files)}")
    print(f"[⚠️] Duplicate paths: {len(duplicates)}")
    print(f"[⚠️] Empty or invalid rows: {len(bad_rows)}")

    # Filter and save cleaned version
    if total_invalid:
        choice = input("\nDo you want to remove invalid entries and save cleaned CSV with durations? (y/n): ").lower()
        if choice == 'y':
            df_clean = df.drop(index=list(total_invalid)).reset_index(drop=True)
            df_clean["duration_sec"] = durations
            df_clean.to_csv(CLEAN_CSV_OUTPUT, index=False)
            print(f"[💾] Cleaned dataset with durations saved to: {CLEAN_CSV_OUTPUT}")
        else:
            print("[ℹ️] No file saved.")
    else:
        df["duration_sec"] = durations
        df.to_csv(CLEAN_CSV_OUTPUT, index=False)
        print(f"[🎉] Dataset was already clean. Saved with durations to: {CLEAN_CSV_OUTPUT}")


if __name__ == "__main__":
    validate_and_augment_dataset()
