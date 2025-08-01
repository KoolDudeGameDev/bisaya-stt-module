import re
from datasets import load_from_disk, DatasetDict

dataset = load_from_disk("data/preprocessed/v1_bisaya")

def clean_text(batch):
    batch["text"] = re.sub(r"[,.?!;:]", "", batch["text"])
    return batch

cleaned_dataset = dataset.map(clean_text)

cleaned_dataset.save_to_disk("data/preprocessed/v1_bisaya_clean")
