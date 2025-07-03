from transformers import Wav2Vec2Processor, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor

# Load your tokenizer
tokenizer = Wav2Vec2CTCTokenizer(
    "vocab.json",
    unk_token="[UNK]",
    pad_token="[PAD]",
    word_delimiter_token="|",
)

# Load the feature extractor
feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1,
    sampling_rate=16000,
    padding_value=0.0,
    do_normalize=True,
    return_attention_mask=True,
)

# Combine into processor
processor = Wav2Vec2Processor(
    feature_extractor=feature_extractor, tokenizer=tokenizer)

# Save processor to a clean directory (no slash at end)
processor.save_pretrained("./processor")
