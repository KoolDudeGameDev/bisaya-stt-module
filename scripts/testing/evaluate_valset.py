# scripts/evaluate_valset.py

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from datasets import load_from_disk
from jiwer import wer
import torch

DATASET_PATH = "data/processed/v1_training_ready_grapheme"
MODEL_DIR = "models/wav2vec2/v1_cebuano"

print("🔁 Loading model and processor...")
processor = Wav2Vec2Processor.from_pretrained(MODEL_DIR)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_DIR)
model.eval()

print("📦 Loading dataset...")
dataset = load_from_disk(DATASET_PATH)
dataset = dataset["train"].train_test_split(test_size=0.1)["test"]

preds, refs = [], []

for sample in dataset:
    input_values = processor(sample["input_values"], return_tensors="pt").input_values
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    pred = processor.batch_decode(predicted_ids)[0]
    preds.append(pred.lower())
    refs.append(sample["sentence"].lower())

error = wer(refs, preds)
print(f"📉 Full WER on test set: {error:.4f}")
