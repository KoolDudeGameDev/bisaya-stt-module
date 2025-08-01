from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2Config
import os

# === Paths ===
PROCESSOR_DIR = "processor/v1_grapheme"
MODEL_SAVE_DIR = "models/wav2vec2-bisaya"

# === Load processor ===
print("🔄 Loading processor...")
processor = Wav2Vec2Processor.from_pretrained(PROCESSOR_DIR)

# === Define model config ===
print("🔧 Creating model config...")
config = Wav2Vec2Config(
    vocab_size=len(processor.tokenizer),
    hidden_size=768,
    num_attention_heads=12,
    num_hidden_layers=12,
    intermediate_size=3072,
    hidden_dropout=0.1,
    activation_dropout=0.1,
    feat_proj_dropout=0.0,
    mask_time_prob=0.05,
    layerdrop=0.1,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    bos_token_id=None,
    eos_token_id=None,
)

# === Initialize model from config ===
print("🧠 Initializing model from scratch...")
model = Wav2Vec2ForCTC(config)

# === Save model and processor ===
print(f"💾 Saving processor and model to: {MODEL_SAVE_DIR}")
model.save_pretrained(MODEL_SAVE_DIR)
processor.save_pretrained(MODEL_SAVE_DIR)

print("✅ Processor and untrained model saved successfully.")
