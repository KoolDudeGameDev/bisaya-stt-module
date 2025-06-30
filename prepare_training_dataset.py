from datasets import load_from_disk, Dataset
from transformers import Wav2Vec2Processor
import torch

dataset = load_from_disk("bisaya-preprocessed-dataset")
processor = Wav2Vec2Processor.from_pretrained("processor/")


def prepare(batch):
    audio = batch["path"]
    batch["input_values"] = processor(
        audio["array"], sampling_rate=16000).input_values[0]
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    return batch


dataset = dataset.map(prepare, remove_columns=dataset.column_names)
dataset.save_to_disk("bisaya-training-ready-dataset")

print("[✅] Saved training-ready dataset")
