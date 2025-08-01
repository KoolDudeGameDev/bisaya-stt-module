from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor

TOKENIZER_DIR = "processor/v1_grapheme"
OUTPUT_DIR = "processor/v1_grapheme"

# === Load tokenizer ===
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(TOKENIZER_DIR)

# === Create feature extractor (standard settings) ===
feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1,
    sampling_rate=16000,
    padding_value=0.0,
    do_normalize=True,
    return_attention_mask=True,
)

# === Wrap processor ===
processor = Wav2Vec2Processor(
    feature_extractor=feature_extractor,
    tokenizer=tokenizer
)

# === Save processor ===
processor.save_pretrained(OUTPUT_DIR)

print(f"✅ Wav2Vec2Processor saved to: {OUTPUT_DIR}")
