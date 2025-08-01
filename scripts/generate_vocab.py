from collections import Counter
import json

corpus_path = "data/final/final_train_manifest.csv"
vocab_path = "tokenizer/v2_grapheme/vocab.json"

counter = Counter()
with open(corpus_path, "r", encoding="utf-8") as f:
    for line in f:
        if "," in line:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                text = parts[1].strip()
                counter.update(list(text))

# Sort by frequency then alphabetically
sorted_vocab = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
tokens = [char for char, _ in sorted_vocab]

# Add special tokens manually
tokens = ["|"] + tokens + ["[UNK]", "[PAD]"]
vocab_dict = {token: idx for idx, token in enumerate(tokens)}

with open(vocab_path, "w", encoding="utf-8") as f:
    json.dump(vocab_dict, f, indent=2)

print("✅ New vocab written to:", vocab_path)
