import json
from datasets import load_from_disk
from pathlib import Path

# === VERSION TAGS ===
DATASET_VERSION = "v1_bisaya"
TOKENIZER_VERSION = "v1_grapheme"

# === Load dataset ===
dataset = load_from_disk(f"data/preprocessed/{DATASET_VERSION}")
if "train" not in dataset:
    raise ValueError("❌ 'train' split not found in DatasetDict.")

# === Extract unique characters and normalize ===
vocab_set = set()
for example in dataset["train"]:
    text = example["text"].strip().lower().replace(" ", "|")
    vocab_set.update(list(text))

# === Build vocab dictionary ===
vocab_list = sorted(vocab_set)
vocab_dict = {char: idx for idx, char in enumerate(vocab_list)}

# === Add special tokens ===
vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)

# === Save tokenizer ===
output_dir = Path(f"processor/{TOKENIZER_VERSION}")
output_dir.mkdir(parents=True, exist_ok=True)
with open(output_dir / "vocab.json", "w", encoding="utf-8") as f:
    json.dump(vocab_dict, f, ensure_ascii=False, indent=2)

print(f"✅ Grapheme tokenizer saved to: {output_dir}")
print(f"🔠 Vocab size: {len(vocab_dict)}")
