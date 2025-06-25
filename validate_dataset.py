import os
import pandas as pd

# Config
CSV_PATH = "bisaya-dataset/bisaya_dataset.csv"
AUDIO_DIR = "bisaya-dataset/audio"
CLEAN_CSV_OUTPUT = "bisaya-dataset/bisaya_dataset_clean.csv"

def validate_dataset():
    if not os.path.exists(CSV_PATH):
        print(f"[❌] CSV not found at {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)

    required_columns = {"path", "text", "category"}
    if not required_columns.issubset(df.columns):
        print(f"[❌] Missing required columns. Found: {df.columns}")
        return

    print(f"\n[📊] Total entries in CSV: {len(df)}")

    seen_paths = set()
    missing_files = []
    duplicates = []
    bad_rows = []

    for idx, row in df.iterrows():
        path = row["path"]
        text = str(row["text"]).strip()
        category = str(row["category"]).strip()

        full_path = os.path.join(AUDIO_DIR, os.path.basename(path))
        is_duplicate = path in seen_paths

        if is_duplicate:
            duplicates.append(idx)
        else:
            seen_paths.add(path)

        if not os.path.exists(full_path):
            missing_files.append(idx)
        elif not path or not text or not category:
            bad_rows.append(idx)

    print(f"[✅] Valid entries: {len(df) - len(missing_files + bad_rows + duplicates)}")
    print(f"[⚠️] Missing audio files: {len(missing_files)}")
    print(f"[⚠️] Duplicated paths: {len(duplicates)}")
    print(f"[⚠️] Empty/malformed rows: {len(bad_rows)}")

    total_invalid = set(missing_files + duplicates + bad_rows)
    if total_invalid:
        choice = input("\nDo you want to remove invalid entries and save a cleaned CSV? (y/n): ").lower()
        if choice == 'y':
            df_clean = df.drop(index=list(total_invalid))
            df_clean.to_csv(CLEAN_CSV_OUTPUT, index=False)
            print(f"[💾] Cleaned CSV saved to: {CLEAN_CSV_OUTPUT}")
        else:
            print("[🔍] No changes made.")
    else:
        print("[🎉] No issues found. Dataset is clean!")

if __name__ == "__main__":
    validate_dataset()
