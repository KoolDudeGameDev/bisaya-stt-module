import pandas as pd
import os

# Define manifest paths
MANIFESTS = {
    "common_voice": "data/raw/manifests/common_voice_cebuano_manifest.csv",
    "own_recording": "data/raw/manifests/my_recordings_manifest.csv",
    "synthetic": "data/synthetic/manifests/manifest_synthetic_cebuano_v1.csv"
}

# Load and label
all_dfs = []
for source, path in MANIFESTS.items():
    if os.path.exists(path):
        df = pd.read_csv(path)
        df["source"] = source
        all_dfs.append(df)
    else:
        print(f"⚠️ Skipped missing: {path}")

# Concatenate and shuffle
merged_df = pd.concat(all_dfs, ignore_index=True)
merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save merged manifest
os.makedirs("data/final", exist_ok=True)
output_path = "data/final/final_bisaya_manifest.csv"
merged_df.to_csv(output_path, index=False)
print(f"✅ Merged manifest saved to: {output_path}")