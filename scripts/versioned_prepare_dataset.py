from datasets import load_dataset
from pathlib import Path
import pandas as pd

# === VERSION TAGS ===
DATASET_VERSION = "v1_bisaya"
INPUT_CSV = "data/final/final_train_manifest.csv"
OUTPUT_DIR = Path(f"data/preprocessed/{DATASET_VERSION}")

# === Normalize paths ===
project_root = Path(".").resolve()
df = pd.read_csv(INPUT_CSV)

def rebase_path(p):
    p = Path(p)
    # Attempt to extract relative path from known subfolder (e.g. 'data/synthetic')
    try:
        idx = p.parts.index("data")
        relative_path = Path(*p.parts[idx:])  # e.g., data/synthetic/v1/audio/file.wav
    except ValueError:
        raise ValueError(f"[❌] Cannot extract relative path from: {p}")

    # Check if that path exists in the current machine
    full = project_root / relative_path
    if not full.exists():
        raise FileNotFoundError(f"[❌] File not found at expected path: {full}")
    
    return str(relative_path)

# Apply path rebasing
df["path"] = df["path"].apply(rebase_path)

# Save temp cleaned CSV
temp_csv = f"data/final/_temp_{DATASET_VERSION}.csv"
df.to_csv(temp_csv, index=False)

# Load and save dataset
dataset = load_dataset(
    "csv",
    data_files={"train": temp_csv},
    delimiter=","
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
dataset.save_to_disk(str(OUTPUT_DIR))

# Cleanup temp
Path(temp_csv).unlink()

print(f"✅ DatasetDict saved to: {OUTPUT_DIR}")
print(f"📄 Total samples: {len(dataset['train'])}")
