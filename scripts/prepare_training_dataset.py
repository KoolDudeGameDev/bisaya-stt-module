import os
import pandas as pd
from pathlib import Path
from datasets import Dataset, DatasetDict, Audio
from transformers import Wav2Vec2Processor
import numpy as np

# === CONFIG ===
MANIFEST_PATH = "data/final/cleaned_train_manifest.csv"
PROCESSOR_DIR = "processor"
PROCESSOR_VERSION = "v1_grapheme"
OUTPUT_VERSION = "v1_training_ready_grapheme"

# === Load processor globally (multiprocessing safe)
processor = None

def prepare(batch):
    global processor
    if processor is None:
        processor = Wav2Vec2Processor.from_pretrained(f"{PROCESSOR_DIR}/{PROCESSOR_VERSION}")

    # Save original path for debugging
    batch["audio_path"] = batch["audio"]["path"]

    # === Audio processing ===
    audio = batch["audio"]
    try:
        input_values = processor(audio["array"], sampling_rate=16000).input_values[0]
    except Exception as e:
        print(f"❌ Audio processing failed for: {batch['audio_path']}")
        print("Error:", e)
        input_values = []

    # Assert audio has valid shape
    if isinstance(audio["array"], np.ndarray):
        if audio["array"].shape[0] < 8000:
            print(f"⚠️ Short audio detected: {batch['audio_path']} ({audio['array'].shape[0]} samples)")

    batch["input_values"] = input_values

    # === Text tokenization ===
    try:
        labels = processor.tokenizer(batch["text"], return_attention_mask=False).input_ids
    except Exception as e:
        print(f"❌ Tokenization failed for: {batch['audio_path']}")
        print("Error:", e)
        labels = []

    if len(labels) == 0:
        print(f"⚠️ Empty label sequence for text: '{batch['text']}' at {batch['audio_path']}")

    batch["labels"] = labels

    return batch


# === Main ===
if __name__ == "__main__":
    print("🔄 Loading and preparing dataset...")

    # Step 1: Load cleaned CSV manifest
    df = pd.read_csv(MANIFEST_PATH)

    # Step 2: Validate columns
    if not {"path", "text"}.issubset(df.columns):
        raise ValueError("❌ Manifest must contain 'path' and 'text' columns.")

    # Step 3: Store original path for traceability
    df["audio_path"] = df["path"]

    # Step 4: Convert to Hugging Face dataset
    dataset = Dataset.from_pandas(df)

    # Step 5: Interpret 'path' column as audio files
    dataset = dataset.cast_column("path", Audio(sampling_rate=16000))

    # Step 6: Rename 'path' to 'audio'
    dataset = dataset.rename_column("path", "audio")

    # Step 7: Preview raw data
    print("🔍 Raw sample BEFORE prepare():")
    raw_sample = dataset[0]
    print(f"🗂 File: {raw_sample['audio']['path']}")
    print(f"📝 Text: {raw_sample['text']}")
    print(f"📏 Waveform shape: {raw_sample['audio']['array'].shape}")
    
    # Step 8: Process dataset
    dataset = dataset.map(
        prepare,
        num_proc=4,
        desc="🔄 Encoding audio & transcript...",
        remove_columns=[]  # Keep all columns for debugging
    )

    # Step 9: Inspect processed sample
    print("🔍 Sample AFTER prepare():")
    sample = dataset[0]
    print(f"🗂 File: {sample['audio_path']}")
    print(f"📝 Text: {sample['text']}")
    print(f"🔡 Labels: {sample['labels']}")
    print(f"🎧 Input values (length): {len(sample['input_values'])}")

    # Step 10: Wrap in DatasetDict and save
    dataset_dict = DatasetDict({"train": dataset})
    output_path = Path(f"data/processed/{OUTPUT_VERSION}")
    output_path.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(output_path)

    # Step 11: Summary
    print(f"✅ Training-ready dataset saved to: {output_path}")
    print(f"📊 Total samples: {len(dataset)}")
