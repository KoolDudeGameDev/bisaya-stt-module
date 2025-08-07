import pandas as pd

# File paths
INPUT_PATH = "data/final/final_train_manifest.csv"
OUTPUT_PATH = "data/final/cleaned_train_manifest.csv"

# Load manifest
df = pd.read_csv(INPUT_PATH)

# Ensure all text is string
df["text"] = df["text"].astype(str)

# Replace | with space
df["text"] = df["text"].str.replace("|", " ", regex=False)

# Normalize spacing and strip whitespace
df["text"] = df["text"].str.replace(r"\s+", " ", regex=True).str.strip()

# Convert text to lowercase
df["text"] = df["text"].str.lower()

# Save cleaned manifest
df.to_csv(OUTPUT_PATH, index=False)
