import os
import re
import torch
import pandas as pd
import soundfile as sf
from dotenv import load_dotenv
from transformers import VitsModel, AutoTokenizer
from glob import glob
import argparse

# ========== ARGPARSE ==========
parser = argparse.ArgumentParser(description="Generate synthetic Cebuano audio from corpus.")
parser.add_argument("--version", required=True, help="Corpus version tag, e.g. 'v3', 'tts_augmented_v1'")
parser.add_argument("--reset", action="store_true", help="Start index from 0 even if audio exists")
args = parser.parse_args()
VERSION = args.version.strip()
# ==============================

# ========== PATH CONFIG ==========
BASE_DIR = f"data/synthetic/{VERSION}"
TEXT_CORPUS = f"data/raw/cebuano_text_corpus_{VERSION}.txt"
SYNTH_DIR = os.path.join(BASE_DIR, "audio")
MANIFEST_PATH = os.path.join(BASE_DIR, f"manifest_{VERSION}.csv")
FAILED_LOG = os.path.join(BASE_DIR, "failed.txt")
SKIPPED_LOG = os.path.join(BASE_DIR, "skipped.txt")
MODEL_ID = "facebook/mms-tts-ceb"
# =================================

# Load Hugging Face token
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if not HF_TOKEN:
    raise ValueError("❌ Missing HUGGINGFACE_TOKEN in .env")

# Ensure output folders exist
os.makedirs(SYNTH_DIR, exist_ok=True)
os.makedirs(BASE_DIR, exist_ok=True)

# Load model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
print("⬇️ Loading MMS Cebuano model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_auth_token=HF_TOKEN)
model = VitsModel.from_pretrained(MODEL_ID, use_auth_token=HF_TOKEN).to(device)
SAMPLE_RATE = model.config.sampling_rate
print(f"✅ Model loaded (sample rate: {SAMPLE_RATE})")

# Determine starting index
start_idx = 0
if not args.reset:
    existing = glob(os.path.join(SYNTH_DIR, f"{VERSION}_synthetic_*.wav"))
    existing_ids = [
        int(re.search(rf"{VERSION}_synthetic_(\d+)\.wav", f).group(1))
        for f in existing
        if re.search(rf"{VERSION}_synthetic_(\d+)\.wav", f)
    ]
    start_idx = max(existing_ids) + 1 if existing_ids else 0

# Load and clean text corpus
if not os.path.exists(TEXT_CORPUS):
    raise FileNotFoundError(f"❌ Corpus not found at: {TEXT_CORPUS}")

with open(TEXT_CORPUS, "r", encoding="utf-8") as f:
    raw_lines = [line.strip() for line in f if line.strip()]

sentences = [s for s in raw_lines if len(s) >= 3 and re.search(r"[a-zA-Z]", s)]
skipped = [s for s in raw_lines if s not in sentences]
if skipped:
    with open(SKIPPED_LOG, "w", encoding="utf-8") as log:
        for line in skipped:
            log.write(f"[invalid-line] {line}\n")
    print(f"⚠️ Skipped {len(skipped)} invalid lines (logged).")

print(f"✅ {len(sentences)} valid sentences to synthesize.")

# Generate synthetic audio
records = []
for i, text in enumerate(sentences):
    idx = start_idx + i
    out_path = os.path.join(SYNTH_DIR, f"{VERSION}_synthetic_{idx:06d}.wav")

    try:
        # Tokenization
        try:
            inputs = tokenizer(text, return_tensors="pt").to(device)
        except Exception:
            with open(FAILED_LOG, "a", encoding="utf-8") as log:
                log.write(f"[tokenizer-error] {text}\n")
            print(f"❌ [tokenizer-error] '{text}'")
            continue

        # Inference
        try:
            with torch.no_grad():
                output = model(**inputs).waveform
        except Exception:
            with open(FAILED_LOG, "a", encoding="utf-8") as log:
                log.write(f"[model-error] {text}\n")
            print(f"❌ [model-error] '{text}'")
            continue

        waveform = output.squeeze().cpu().numpy()
        duration_sec = len(waveform) / SAMPLE_RATE
        max_amplitude = waveform.max()

        # Quality filter
        if duration_sec < 1.0:
            with open(FAILED_LOG, "a", encoding="utf-8") as log:
                log.write(f"[short-duration] {text}\n")
            print(f"⛔ [short-duration] '{text}'")
            continue
        if duration_sec > 10.0:
            with open(FAILED_LOG, "a", encoding="utf-8") as log:
                log.write(f"[long-duration] {text}\n")
            print(f"⛔ [long-duration] '{text}'")
            continue
        if max_amplitude < 0.05:
            with open(FAILED_LOG, "a", encoding="utf-8") as log:
                log.write(f"[low-amplitude] {text}\n")
            print(f"⛔ [low-amplitude] '{text}'")
            continue

        # Save audio
        sf.write(out_path, waveform, SAMPLE_RATE)

        records.append({
            "path": os.path.abspath(out_path),
            "text": text.lower(),
            "source": f"synthetic_{VERSION}",
            "duration_sec": duration_sec
        })
        print(f"🎧 [{idx:06d}] Generated: '{text}'")

    except Exception as e:
        with open(FAILED_LOG, "a", encoding="utf-8") as log:
            log.write(f"[unknown-error] {text} — {e}\n")
        print(f"❌ [unknown-error] '{text}' — {e}")

# Save manifest
if records:
    df = pd.DataFrame(records)
    if not args.reset and os.path.exists(MANIFEST_PATH):
        old_df = pd.read_csv(MANIFEST_PATH)
        df = pd.concat([old_df, df], ignore_index=True)
    df.to_csv(MANIFEST_PATH, index=False)
    print(f"✅ Manifest saved to: {MANIFEST_PATH} with {len(df)} total entries.")
else:
    print("⚠️ No audio was generated.")
