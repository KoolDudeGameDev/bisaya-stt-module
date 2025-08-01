import argparse
import json
from datasets import load_from_disk
from pathlib import Path

# === CLI Args ===
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_version", required=True)
parser.add_argument("--output_dir", required=True)
args = parser.parse_args()

DATASET_VERSION = args.dataset_version
OUTPUT_DIR = Path(args.output_dir)

# === Load cleaned dataset ===
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
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_DIR / "vocab.json", "w", encoding="utf-8") as f:
    json.dump(vocab_dict, f, ensure_ascii=False, indent=2)

print(f"✅ Grapheme tokenizer saved to: {OUTPUT_DIR}")
print(f"🔠 Vocab size: {len(vocab_dict)}")
