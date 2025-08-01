from transformers import Wav2Vec2CTCTokenizer
from pathlib import Path

# === Paths ===
VOCAB_PATH = "processor/v1_grapheme/vocab.json"
OUTPUT_DIR = "processor/v1_grapheme"

# === Export tokenizer ===
tokenizer = Wav2Vec2CTCTokenizer(
    vocab_file=VOCAB_PATH,
    unk_token="[UNK]",
    pad_token="[PAD]",
    word_delimiter_token="|",
    lowercase=True,
)

tokenizer.save_pretrained(OUTPUT_DIR)

print(f"✅ Exported tokenizer to: {OUTPUT_DIR}")
