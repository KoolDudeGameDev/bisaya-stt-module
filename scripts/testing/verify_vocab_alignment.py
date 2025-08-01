# scripts/verify_vocab_alignment.py

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
MODEL_DIR = "models/wav2vec2-cebuano"

model = Wav2Vec2ForCTC.from_pretrained(MODEL_DIR)
processor = Wav2Vec2Processor.from_pretrained(MODEL_DIR)

vocab_len = len(processor.tokenizer)
head_len = model.lm_head.out_features

assert vocab_len == head_len, f"❌ Mismatch: tokenizer={vocab_len}, model_head={head_len}"
print(f"✅ Vocab check passed: tokenizer={vocab_len}, model_head={head_len}")
