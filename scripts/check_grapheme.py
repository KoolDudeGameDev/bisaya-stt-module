import json
from datasets import load_from_disk
from transformers import Wav2Vec2Processor

# Load tokenizer and dataset
processor = Wav2Vec2Processor.from_pretrained("processor/v1_grapheme")
dataset = load_from_disk("data/processed/v1_training_ready_grapheme")

# Load vocab
with open("processor/v1_grapheme/vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)
vocab_tokens = set(vocab.keys())

# Collect all label graphemes
label_graphemes = set()
for sample in dataset["train"]:
    decoded = processor.tokenizer.decode(sample["labels"], skip_special_tokens=False)
    label_graphemes.update(list(decoded))

# Check missing
missing = label_graphemes - vocab_tokens
print(f"🔍 Unique graphemes in dataset: {sorted(label_graphemes)}")
print(f"✅ Vocab tokens: {sorted(vocab_tokens)}")

if missing:
    print(f"❌ Missing graphemes in vocab: {missing}")
else:
    print("✅ All label graphemes are covered by the tokenizer vocab.")
