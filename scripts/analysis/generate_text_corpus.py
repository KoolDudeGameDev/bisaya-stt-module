import pandas as pd
import os
import argparse

# ========== ARGPARSE ==========
parser = argparse.ArgumentParser(description="Generate clean Cebuano corpus from CSV file.")
parser.add_argument("--csv", required=True, help="Path to CSV file (e.g. cebuano_ipa_dataset.csv)")
parser.add_argument("--version", required=True, help="Corpus version tag (e.g. v2, ipa_set_1)")
args = parser.parse_args()

CSV_PATH = args.csv
VERSION = args.version.strip()
OUT_PATH = f"data/raw/cebuano_text_corpus_{VERSION}.txt"
# ==============================

# ========== LOAD CSV ==========
try:
    df = pd.read_csv(
        CSV_PATH,
        encoding="utf-8",  # ✅ Fix: Use UTF-8 for IPA compatibility
        quoting=1,
        on_bad_lines="warn",
        engine="python"
    )
except Exception as e:
    print(f"❌ Failed to load CSV: {e}")
    raise

print(f"✅ Loaded CSV: {CSV_PATH}")
# ==============================

# ========== EXTRACT TEXT COLUMN ==========
# Use `word` column since this dataset uses that label for Cebuano words
if "word" not in df.columns:
    raise ValueError(f"❌ 'word' column not found. Found: {df.columns}")

texts = df["word"].dropna().astype(str).str.strip()
texts = texts[texts != ""].unique()
# =========================================

# ========== SAVE TO OUTPUT ==========
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

with open(OUT_PATH, "w", encoding="utf-8") as f:
    for line in texts:
        f.write(line + "\n")

print(f"✅ Saved {len(texts)} Cebuano lines to: {OUT_PATH}")
# ====================================
