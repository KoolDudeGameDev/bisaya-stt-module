# prepare_hf_dataset.py

from datasets import Dataset, Audio
import pandas as pd

# Load CSV manifest
df = pd.read_csv("final_bisaya_manifest.csv")

# Optionally inspect
print(df.head())

# Create Dataset
dataset = Dataset.from_pandas(df)

# Cast 'path' column to Audio
dataset = dataset.cast_column("path", Audio(sampling_rate=16000))

# Save to disk
dataset.save_to_disk("bisaya-training-ready-dataset")

print("HF dataset saved to bisaya-training-ready-dataset")
