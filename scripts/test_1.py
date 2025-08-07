import os
import pandas as pd
import shutil

# === Config ===
CSV_PATH = "data/raw/real_dataset.csv"  # path to the CSV file
AUDIO_ROOT = "data/raw/audio/real_v1"   # folder containing .wavs directly
BACKUP = True
CATEGORY = "phrase"
SPEAKER = "kyle"
VERSION = "real_v1"

# === Backup CSV ===
if BACKUP:
    backup_csv = CSV_PATH.replace(".csv", "_backup.csv")
    shutil.copyfile(CSV_PATH, backup_csv)
    print(f"✅ Backup created at: {backup_csv}")

# === Load and filter ===
df = pd.read_csv(CSV_PATH)
df = df[df['version'] == VERSION].reset_index(drop=True)

# === Rename + reindex ===
new_rows = []
for i, row in enumerate(df.itertuples(), start=1):
    new_index = f"{i:04d}"
    
    # Extract only filename, ignore any previous subfolder
    old_filename = os.path.basename(row.path.replace("\\", "/"))
    old_file = os.path.join(AUDIO_ROOT, old_filename)

    new_filename = f"{VERSION}_{SPEAKER}_{new_index}.wav"
    new_rel_path = f"audio/{VERSION}/{new_filename}"
    new_file = os.path.join(AUDIO_ROOT, new_filename)

    if os.path.exists(old_file):
        os.rename(old_file, new_file)
    else:
        print(f"⚠️ Missing file: {old_file} — skipping")
        continue

    new_rows.append({
        "path": new_rel_path.replace("/", "\\"),
        "text": row.text,
        "category": CATEGORY,
        "speaker": SPEAKER,
        "version": VERSION,
        "timestamp": row.timestamp
    })

# === Save cleaned CSV ===
df_out = pd.DataFrame(new_rows)
df_out.to_csv(CSV_PATH, index=False)
print(f"✅ Reindexed and cleaned CSV saved to: {CSV_PATH}")
print(f"🔁 Total valid entries reindexed: {len(df_out)}")
