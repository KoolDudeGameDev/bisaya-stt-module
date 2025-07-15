from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor
from pathlib import Path

# === VERSION TAGS ===
TOKENIZER_VERSION = "v1_grapheme"
PROCESSOR_VERSION = TOKENIZER_VERSION  # keep same version for now

# === Load tokenizer ===
tokenizer = Wav2Vec2CTCTokenizer(
    vocab_file=str(Path(f"tokenizer/{TOKENIZER_VERSION}/vocab.json")),
    unk_token="[UNK]",
    pad_token="[PAD]",
    word_delimiter_token="|"
)

# === Load feature extractor ===
feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1,
    sampling_rate=16000,
    padding_value=0.0,
    do_normalize=True,
    return_attention_mask=True,
)

# === Bundle into processor ===
processor = Wav2Vec2Processor(
    feature_extractor=feature_extractor,
    tokenizer=tokenizer,
)

# === Save processor ===
output_dir = Path(f"processor/{PROCESSOR_VERSION}")
output_dir.mkdir(parents=True, exist_ok=True)
processor.save_pretrained(str(output_dir))

print(f"✅ Processor saved to: {output_dir}")
