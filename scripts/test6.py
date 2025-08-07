from datasets import load_from_disk

dataset = load_from_disk("data/processed/v1_training_ready_grapheme")["train"]

sample = dataset[0]
print("Text:", sample["text"])
print("Audio path:", sample["audio_path"])
print("Labels:", sample["labels"])
print("Input values length:", len(sample["input_values"]))
