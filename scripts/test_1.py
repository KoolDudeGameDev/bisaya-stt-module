from transformers import Wav2Vec2CTCTokenizer

tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("models/tokenizer-v1-grapheme")
print(tokenizer.tokenize("ganahan ko moorder og torta"))
