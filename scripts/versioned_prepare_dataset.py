# scripts/versioned_prepare_dataset.py
from datasets import load_dataset
from pathlib import Path

# === VERSION TAGS ===
DATASET_VERSION = "v1_bisaya"
INPUT_CSV = "data/final/final_train_manifest.csv"
OUTPUT_DIR = Path(f"data/preprocessed/{DATASET_VERSION}")

# === Load CSV as train split ===
dataset = load_dataset(
    "csv",
    data_files={"train": INPUT_CSV},
    delimiter=","
)

# === Save as versioned Hugging Face DatasetDict ===
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
dataset.save_to_disk(str(OUTPUT_DIR))

print(f"✅ DatasetDict saved to: {OUTPUT_DIR}")
print(f"📄 Total samples: {len(dataset['train'])}")
