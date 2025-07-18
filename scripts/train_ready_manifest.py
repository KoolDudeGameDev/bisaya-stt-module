# ========== GUIDE ==========
# Compile and clean all synthetic and raw datasets into a training-ready manifest.
# Usage: python train_ready_manifest.py --include_clean
# Synthetic audio will be cleaned
# Usage: python train_ready_manifest.py
# ==============================

import os
import pandas as pd
import soundfile as sf
from glob import glob
import argparse
from tqdm import tqdm

# ========== ARGPARSE ==========
parser = argparse.ArgumentParser(description="Compile and clean all synthetic and raw datasets into a training-ready manifest.")
parser.add_argument("--include_clean", action="store_true", help="Include raw bisaya_dataset_clean.csv")
parser.add_argument("--min_duration", type=float, default=1.0, help="Minimum valid audio duration (sec)")
parser.add_argument("--max_duration", type=float, default=10.0, help="Maximum valid audio duration (sec)")
parser.add_argument("--output", default="data/final/final_train_manifest.csv", help="Output manifest path")
args = parser.parse_args()

# ========== CONFIG ==========
RAW_CLEAN_PATH = "data/raw/bisaya-dataset/bisaya_dataset_clean.csv"
SYNTH_BASE_DIR = "data/synthetic"
COLUMNS_REQUIRED = {"path", "text"}
OUTPUT_PATH = args.output

# ========== HELPERS ==========
def get_duration_sec(path):
    try:
        return round(sf.info(path).duration, 3)
    except Exception:
        return None

def load_and_clean_manifest(path, source_name, version_prefix=None):
    df = pd.read_csv(path)

    if not COLUMNS_REQUIRED.issubset(df.columns):
        print(f"⚠️ Skipping {path}: missing required columns.")
        return pd.DataFrame()

    # Clean text and duration
    df["text"] = df["text"].astype(str).str.strip().str.lower()
    df["text"] = df["text"].str.replace(" ", "|", regex=False)  # CTC-compatible space


    # Ensure duration exists or recalculate
    if "duration_sec" not in df.columns or df["duration_sec"].isnull().any():
        print(f"ℹ️ Recalculating durations in {source_name}...")
        df["duration_sec"] = df["path"].apply(lambda p: get_duration_sec(p))

    # Filter durations
    df = df[(df["duration_sec"] >= args.min_duration) & (df["duration_sec"] <= args.max_duration)]

    # Ensure path is absolute
    df["path"] = df["path"].apply(lambda p: os.path.abspath(p))

    # Filter out missing files
    df = df[df["path"].apply(os.path.exists)]

    # Drop duplicates
    df = df.drop_duplicates(subset=["text", "path"])

    # Assign source
    df["source"] = source_name

    # Rebuild filename column (optional)
    if version_prefix:
        df["filename"] = df["path"].apply(lambda p: os.path.basename(p))
        df["filename"] = df["filename"].apply(lambda f: f"{version_prefix}_{f}" if not f.startswith(version_prefix) else f)

    print(f"📊 {source_name}: Loaded {len(df)} valid rows after cleaning.")

    return df.reset_index(drop=True)

# ========== LOAD SYNTHETIC MANIFESTS ==========
print("🔍 Scanning synthetic dataset versions...")

synthetic_dfs = []
version_folders = sorted([d for d in os.listdir(SYNTH_BASE_DIR) if os.path.isdir(os.path.join(SYNTH_BASE_DIR, d)) and d.startswith("v")])

for version in version_folders:
    manifest_path = os.path.join(SYNTH_BASE_DIR, version, f"manifest_{version}.csv")
    if not os.path.exists(manifest_path):
        print(f"⚠️ Skipping {version} — cleaned manifest not found.")
        continue

    print(f"✅ Loading {version}...")
    df_version = load_and_clean_manifest(manifest_path, source_name=f"synthetic_{version}", version_prefix=version)
    if not df_version.empty:
        synthetic_dfs.append(df_version)

# ========== LOAD RAW CLEANED DATA ==========
raw_df = pd.DataFrame()
if args.include_clean and os.path.exists(RAW_CLEAN_PATH):
    print("✅ Loading raw clean Bisaya dataset...")
    raw_df = load_and_clean_manifest(RAW_CLEAN_PATH, source_name="clean_corpus")
else:
    print("⚠️ Skipping clean corpus. Use --include_clean to include it.")

# ========== COMBINE & SHUFFLE ==========
all_dfs = synthetic_dfs + ([raw_df] if not raw_df.empty else [])
if not all_dfs:
    raise RuntimeError("❌ No datasets were loaded. Aborting.")

final_df = pd.concat(all_dfs, ignore_index=True)
final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

# ========== EXPORT ==========
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
final_df.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Final manifest saved to: {OUTPUT_PATH}")
print(f"🎧 Total audio samples: {len(final_df)}")
print(f"🔗 Included sources: {final_df['source'].nunique()} ({final_df['source'].unique().tolist()})")
# ========== END ==========