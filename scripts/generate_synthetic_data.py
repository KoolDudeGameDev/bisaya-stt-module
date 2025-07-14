import os
import re
import torch
import pandas as pd
import soundfile as sf
from dotenv import load_dotenv
from transformers import VitsModel, AutoTokenizer
from glob import glob

# ========== CONFIG ==========
TEXT_CORPUS = "data/raw/cebuano_text_corpus_extra.txt"
SYNTH_DIR = "data/synthetic/audio"
MANIFEST_PATH = "data/synthetic/manifests/manifest_synthetic_cebuano_v1.csv"
FAILED_LOG = "data/synthetic/logs/failed_synthetic_lines.txt"
SKIPPED_LOG = "data/synthetic/logs/skipped_synthetic_lines.txt"
MODEL_ID = "facebook/mms-tts-ceb"
# ============================

# Load Hugging Face token
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if not HF_TOKEN:
    raise ValueError("❌ Missing HUGGINGFACE_TOKEN in .env")

# Ensure output folders exist
os.makedirs(SYNTH_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
os.makedirs(os.path.dirname(FAILED_LOG), exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and tokenizer
print("⬇️ Loading MMS Cebuano model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_auth_token=HF_TOKEN)
model = VitsModel.from_pretrained(MODEL_ID, use_auth_token=HF_TOKEN).to(device)
SAMPLE_RATE = model.config.sampling_rate
print(f"✅ Model loaded (sample rate: {SAMPLE_RATE})")

# Determine starting index
existing = glob(os.path.join(SYNTH_DIR, "synthetic_*.wav"))
existing_ids = [int(re.search(r"synthetic_(\d+)\.wav", f).group(1)) for f in existing if re.search(r"synthetic_(\d+)\.wav", f)]
start_idx = max(existing_ids) + 1 if existing_ids else 0

# Load and clean text corpus
with open(TEXT_CORPUS, "r", encoding="utf-8") as f:
    raw_lines = [line.strip() for line in f if line.strip()]

sentences = [s for s in raw_lines if len(s) >= 3 and re.search(r"[a-zA-Z]", s)]
skipped = [s for s in raw_lines if s not in sentences]
if skipped:
    with open(SKIPPED_LOG, "w", encoding="utf-8") as log:
        log.write("\n".join(skipped))
    print(f"⚠️ Skipped {len(skipped)} invalid lines (logged).")

print(f"✅ {len(sentences)} valid sentences to synthesize.")

# Generate synthetic audio
records = []
for i, text in enumerate(sentences):
    idx = start_idx + i
    out_path = os.path.join(SYNTH_DIR, f"synthetic_{idx}.wav")

    try:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model(**inputs).waveform

        waveform = output.squeeze().cpu().numpy()
        duration_sec = len(waveform) / SAMPLE_RATE

        if duration_sec < 1.0 or duration_sec > 10.0 or waveform.max() < 0.05:
            print(f"⛔ Skipping due to quality: {text}")
            continue

        sf.write(out_path, waveform, SAMPLE_RATE)

        records.append({
            "path": out_path,
            "text": text.lower(),
            "source": "synthetic",
            "duration_sec": duration_sec
        })
        print(f"🎧 [{idx}] Generated: '{text}'")

    except Exception as e:
        print(f"❌ Error on '{text}': {e}")
        with open(FAILED_LOG, "a", encoding="utf-8") as log:
            log.write(f"{text}\n")

# Write manifest
if records:
    df = pd.DataFrame(records)
    if os.path.exists(MANIFEST_PATH):
        old_df = pd.read_csv(MANIFEST_PATH)
        df = pd.concat([old_df, df], ignore_index=True)
    df.to_csv(MANIFEST_PATH, index=False)
    print(f"✅ Manifest updated: {len(records)} new entries.")
else:
    print("⚠️ No audio generated.")
