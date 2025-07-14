from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2Processor

tokenizer = Wav2Vec2CTCTokenizer(
    "vocab.json",
    unk_token="[UNK]",
    pad_token="[PAD]",
    word_delimiter_token="|"
)

tokenizer.save_pretrained("tokenizer/")

processor = Wav2Vec2Processor(
    feature_extractor=None,
    tokenizer=tokenizer
)

processor.save_pretrained("processor/")
