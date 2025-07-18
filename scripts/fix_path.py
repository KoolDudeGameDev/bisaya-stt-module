import pandas as pd
import os
from glob import glob

print("🔧 Normalizing paths inside manifest CSVs...")

base_dir = os.path.abspath(".")

for manifest in glob("data/synthetic/*/manifest_*.csv"):
    df = pd.read_csv(manifest)
    if "path" not in df.columns:
        print(f"❌ {manifest} missing 'path' column.")
        continue

    def fix_path(p):
        # Get filename only
        filename = os.path.basename(p.strip())
        # Derive correct full path based on folder
        folder = os.path.dirname(manifest).replace("\\", "/")
        full_path = os.path.join(folder, "audio", filename)
        return os.path.relpath(full_path, base_dir).replace("\\", "/")

    df["path"] = df["path"].apply(fix_path)
    df.to_csv(manifest, index=False)
    print(f"✅ Fixed: {manifest}")
