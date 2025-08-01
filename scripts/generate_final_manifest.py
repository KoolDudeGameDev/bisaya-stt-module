# scripts/generate_final_manifest.py

import pandas as pd
from pathlib import Path

# === Config ===
RAW_BASE_DIR = Path("data/raw")  # this should align with the base of the path values in your CSV
INPUT_CSV = RAW_BASE_DIR / "real_dataset.csv"
OUTPUT_MANIFEST = Path("data/final/final_train_manifest.csv")

# === Load CSV ===
df = pd.read_csv(INPUT_CSV)

# === Adjust column name if needed ===
if "path" in df.columns and "filename" not in df.columns:
    df = df.rename(columns={"path": "filename"})

# === Check required columns ===
required_cols = {"filename", "text"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"❌ real_dataset.csv must contain columns: {required_cols}")

# === Resolve full path ===
def resolve_audio_path(relative_path):
    audio_path = RAW_BASE_DIR / relative_path
    if not audio_path.exists():
        raise FileNotFoundError(f"[❌] Missing audio file: {audio_path}")
    return str(audio_path.resolve())

df["path"] = df["filename"].apply(resolve_audio_path)

# === Reorder & save ===
final_df = df[["path", "text"]]
OUTPUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
final_df.to_csv(OUTPUT_MANIFEST, index=False)

print(f"✅ Final manifest saved: {OUTPUT_MANIFEST}")
print(f"📄 Total samples: {len(final_df)}")
