# scripts/prepare_training_dataset.py

import os
import torchaudio
from datasets import load_from_disk, Audio
from transformers import Wav2Vec2Processor

# === VERSION TAGS ===
DATASET_VERSION = "v1_bisaya"
PROCESSOR_VERSION = "v1_grapheme"
OUTPUT_VERSION = "v1_training_ready"

# === Global processor object (used by child processes) ===
processor = None

# === Prepare audio + transcription ===
def prepare(batch):
    global processor
    if processor is None:
        processor = Wav2Vec2Processor.from_pretrained(f"processor/{PROCESSOR_VERSION}")

    # Audio data is automatically loaded into batch["path"]["array"]
    audio = batch["path"]

    # Convert audio to input_values
    batch["input_values"] = processor(
        audio["array"], sampling_rate=16000
    ).input_values[0]

    # Convert text to token IDs
    batch["labels"] = processor.tokenizer(
        batch["text"], return_attention_mask=False
    ).input_ids

    return batch

# === Entry point (Windows-safe multiprocessing) ===
if __name__ == "__main__":
    # Load dataset
    dataset = load_from_disk(f"data/preprocessed/{DATASET_VERSION}")

    # Cast audio column to ensure automatic audio loading
    dataset = dataset.cast_column("path", Audio(sampling_rate=16000))

    # Apply transformation
    prepared = dataset.map(
        prepare,
        remove_columns=dataset["train"].column_names,
        num_proc=4  # Use multiprocessing
    )

    # Save processed dataset
    output_dir = f"data/processed/{OUTPUT_VERSION}"
    os.makedirs(output_dir, exist_ok=True)
    prepared.save_to_disk(output_dir)

    print(f"[✅] Training-ready dataset saved to: {output_dir}")
