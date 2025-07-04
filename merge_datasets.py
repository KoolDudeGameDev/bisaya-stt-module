# merge_datasets.py

import pandas as pd

# Load individual manifests
cv_df = pd.read_csv("common_voice_cebuano_manifest.csv")
my_df = pd.read_csv("my_recordings_manifest.csv")
synth_df = pd.read_csv("synthetic_manifest.csv")

# Add source labels
cv_df["source"] = "common_voice"
my_df["source"] = "own_recording"
synth_df["source"] = "synthetic"

# Concatenate
final_df = pd.concat([cv_df, my_df, synth_df], ignore_index=True)

# Shuffle rows
final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save final manifest
final_df.to_csv("final_bisaya_manifest.csv", index=False)

print("Merged manifest saved as final_bisaya_manifest.csv")
