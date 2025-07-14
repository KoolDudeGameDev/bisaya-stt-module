import json
from datasets import load_from_disk

# Load your dataset
dataset = load_from_disk("data/preprocessed/bisaya")

# Build vocabulary from your transcripts
vocab = set()
for t in dataset["text"]:
    vocab.update(list(t.lower()))

vocab = sorted(list(vocab))
vocab_dict = {c: i for i, c in enumerate(vocab)}
vocab_dict["|"] = vocab_dict[" "]  # replace space with special token
del vocab_dict[" "]
vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)

os.makedirs("tokenizer", exist_ok=True)
with open("tokenizer/vocab.json", "w") as f:
    json.dump(vocab_dict, f)

print("[✅] Generated tokenizer/vocab.json")
