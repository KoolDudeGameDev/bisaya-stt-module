import pandas as pd

# Load manifest
df = pd.read_csv("data/final/final_train_manifest.csv")

# Replace | with space
df["text"] = df["text"].str.replace("|", " ", regex=False)

# Optional: strip multiple spaces or leading/trailing whitespace
df["text"] = df["text"].str.replace(r"\s+", " ", regex=True).str.strip()

# Save to a new clean manifest
df.to_csv("data/final/final_train_manifest_clean.csv", index=False)
