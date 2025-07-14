import os
import pandas as pd
from glob import glob
from tqdm import tqdm
import shutil

# ===== CONFIG =====
SYNTH_DIR = "data/synthetic/audio"
CLEAN_MANIFEST = "data/synthetic/manifests/manifest_synthetic_cleaned.csv"
RENAMED_MANIFEST = "data/synthetic/manifests/manifest_synthetic_renamed.csv"
# ===================

# Validate paths
if not os.path.exists(CLEAN_MANIFEST):
    raise FileNotFoundError(f"❌ Cleaned manifest not found: {CLEAN_MANIFEST}")

if not os.path.exists(SYNTH_DIR):
    raise FileNotFoundError(f"❌ Synthetic audio directory not found: {SYNTH_DIR}")

# Load manifest
df = pd.read_csv(CLEAN_MANIFEST)
used_files = set(os.path.abspath(p) for p in df["path"])

# Find all .wav files in synthetic folder
all_wavs = glob(os.path.join(SYNTH_DIR, "*.wav"))
all_wavs_abs = [os.path.abspath(p) for p in all_wavs]

# === PART 1: REMOVE UNUSED FILES ===
unused = [p for p in all_wavs_abs if p not in used_files]

if not unused:
    print("✅ No unused .wav files found.")
else:
    print(f"🧹 Found {len(unused)} unused .wav files. Deleting...")
    for f in tqdm(unused, desc="Deleting unused"):
        os.remove(f)
    print("✅ Unused files removed.")

# === PART 2: RENAME USED FILES SEQUENTIALLY AND UPDATE MANIFEST ===

print("🔁 Renaming used files sequentially and updating manifest...")

new_paths = []
for i, row in tqdm(df.iterrows(), total=len(df), desc="Renaming files"):
    old_path = os.path.abspath(row["path"])
    new_filename = f"synthetic_{i:05}.wav"
    new_path = os.path.join(SYNTH_DIR, new_filename)

    # Rename only if the name differs
    if old_path != os.path.abspath(new_path):
        shutil.move(old_path, new_path)

    new_paths.append(os.path.abspath(new_path))

# Update manifest paths
df["path"] = new_paths
df.to_csv(RENAMED_MANIFEST, index=False)

print(f"✅ Renamed all used files.")
print(f"📄 Updated manifest saved to: {RENAMED_MANIFEST}")
