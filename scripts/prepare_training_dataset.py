# scripts/prepare_training_dataset.py

import os
import pandas as pd
import torchaudio
from datasets import Dataset, DatasetDict, Audio
from transformers import Wav2Vec2Processor
from pathlib import Path

# === VERSION TAGS ===
MANIFEST_PATH = "data/final/cleaned_train_manifest.csv"
PROCESSOR_VERSION = "v1_grapheme"
OUTPUT_VERSION = "v1_training_ready_grapheme"

# === Load processor globally for multiprocessing ===
processor = None

# === Sample preparation function ===
def prepare(batch):
    global processor
    if processor is None:
        processor = Wav2Vec2Processor.from_pretrained(f"processor/{PROCESSOR_VERSION}")

    audio = batch["path"]

    # Extract audio input values
    batch["input_values"] = processor(
        audio["array"], sampling_rate=16000
    ).input_values[0]

    # Encode text to label token IDs
    batch["labels"] = processor.tokenizer(
        batch["text"], return_attention_mask=False
    ).input_ids

    return batch

# === Entry point ===
if __name__ == "__main__":
    # Step 1: Load CSV
    df = pd.read_csv(MANIFEST_PATH).rename(columns={"rawpath": "path"})

    if not {"path", "text"}.issubset(df.columns):
        raise ValueError("❌ Manifest must contain 'path' and 'text' columns.")

    # Step 2: Convert to Hugging Face Dataset
    dataset = Dataset.from_pandas(df[["path", "text"]])

    # Step 3: Ensure audio loading
    dataset = dataset.cast_column("path", Audio(sampling_rate=16000))

    # Step 4: Tokenize & preprocess
    dataset = dataset.map(
        prepare,
        remove_columns=dataset.column_names,
        num_proc=4,
        desc="🔄 Preparing dataset...",
    )

    # Step 5: Save as DatasetDict
    dataset_dict = DatasetDict({"train": dataset})

    output_path = Path(f"data/processed/{OUTPUT_VERSION}")
    output_path.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(output_path)

    print(f"✅ Training-ready dataset saved to: {output_path}")
