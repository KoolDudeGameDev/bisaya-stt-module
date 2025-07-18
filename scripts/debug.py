from datasets import load_from_disk
from transformers import Wav2Vec2Processor

dataset = load_from_disk("data/processed/v1_training_ready")
processor = Wav2Vec2Processor.from_pretrained("processor/v1_grapheme")

unk_count = 0
for sample in dataset["train"]:
    decoded = processor.tokenizer.decode(sample["labels"], skip_special_tokens=False)
    if "[UNK]" in decoded:
        unk_count += 1

print(f"🔥 Samples with [UNK]: {unk_count} / {len(dataset['train'])}")
