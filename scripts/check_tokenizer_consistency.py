# scripts/check_tokenizer_consistency.py

from datasets import load_from_disk
from transformers import Wav2Vec2Processor
from pathlib import Path
import json

# === CONFIG ===
DATASET_PATH = "data/preprocessed/v1_bisaya"  # FIXED: use preprocessed dataset
PROCESSOR_PATH = "processor/v1_grapheme"

# === Load dataset and processor ===
print("🔍 Loading dataset and processor...")
dataset = load_from_disk(DATASET_PATH)
processor = Wav2Vec2Processor.from_pretrained(PROCESSOR_PATH)

# === Load vocab manually to inspect raw chars ===
vocab_path = Path(PROCESSOR_PATH) / "vocab.json"
with open(vocab_path, "r", encoding="utf-8") as f:
    vocab = json.load(f)

# === Get reverse vocab for decoding verification ===
inv_vocab = {v: k for k, v in vocab.items()}

# === Extract unique characters from dataset text ===
print("🔠 Extracting graphemes from dataset...")
all_chars = set()
for example in dataset["train"]:
    all_chars.update(list(example["text"]))

missing_chars = sorted([c for c in all_chars if c not in vocab])
if missing_chars:
    print(f"❌ Missing graphemes in tokenizer vocab: {missing_chars}")
else:
    print("✅ All graphemes in dataset exist in tokenizer vocab.")

# === Check for [UNK]s after decoding labels ===
print("🔍 Decoding labels to check for [UNK]s...")
unk_count = 0
total = 0

# You need to prepare label tokens first
processor = Wav2Vec2Processor.from_pretrained(PROCESSOR_PATH)

# Simulate label encoding to catch issues before full processing
for example in dataset["train"]:
    label_ids = processor.tokenizer(example["text"], return_attention_mask=False).input_ids
    decoded = processor.tokenizer.decode(label_ids, skip_special_tokens=False)
    if "[UNK]" in decoded:
        unk_count += 1
    total += 1

if unk_count == 0:
    print(f"✅ No [UNK] tokens found when decoding {total} label sequences.")
else:
    print(f"❌ {unk_count} / {total} label sequences contain [UNK]s.")

print("🔁 Verification complete.")
