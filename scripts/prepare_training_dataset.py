from datasets import load_from_disk
from transformers import Wav2Vec2Processor
import torch

# Load processed dataset and processor
dataset = load_from_disk("data/preprocessed/bisaya")
processor = Wav2Vec2Processor.from_pretrained("processor/")

# Prepare audio/text into model format
def prepare(batch):
    audio = batch["path"]
    batch["input_values"] = processor(
        audio["array"], sampling_rate=16000).input_values[0]
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    return batch

# Apply transformation and save
prepared = dataset.map(prepare, remove_columns=dataset.column_names)
os.makedirs("data/processed", exist_ok=True)
prepared.save_to_disk("data/processed/training_ready")

print("[✅] Saved processed dataset to data/processed/training_ready")
