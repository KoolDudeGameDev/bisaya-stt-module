# scripts/prepare_tokenizer.py
import json
from datasets import load_from_disk
from pathlib import Path

# === VERSION TAGS ===
DATASET_VERSION = "v1_bisaya"
TOKENIZER_VERSION = "v1_grapheme"

# === Load preprocessed DatasetDict ===
dataset = load_from_disk(f"data/preprocessed/{DATASET_VERSION}")

if "train" not in dataset:
    raise ValueError("❌ 'train' split not found. Ensure it's saved as a DatasetDict.")

# === Extract unique characters from 'text' column ===
vocab_set = set()
for example in dataset["train"]:
    vocab_set.update(list(example["text"].strip().lower()))

# === Sort and build vocab ===
vocab_list = sorted(list(vocab_set))
vocab_dict = {char: idx for idx, char in enumerate(vocab_list)}

# === Add CTC-special tokens ===
if " " in vocab_dict:
    vocab_dict["|"] = vocab_dict.pop(" ")
vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)

# === Save vocab to versioned tokenizer path ===
output_dir = Path(f"tokenizer/{TOKENIZER_VERSION}")
output_dir.mkdir(parents=True, exist_ok=True)
with open(output_dir / "vocab.json", "w", encoding="utf-8") as f:
    json.dump(vocab_dict, f, ensure_ascii=False, indent=2)

print(f"✅ Grapheme tokenizer saved to: {output_dir}")
print(f"🔠 Vocab size: {len(vocab_dict)}")
