# prepare_synthetic_cebuano.py

from TTS.api import TTS
import os
import pandas as pd

# Output dir
SYNTH_DIR = "synthetic_bisaya_audio"
os.makedirs(SYNTH_DIR, exist_ok=True)

# Load lines from text corpus
with open("cebuano_text_corpus.txt", "r", encoding="utf-8") as f:
    sentences = [line.strip() for line in f.readlines() if line.strip()]

print(f"Loaded {len(sentences)} sentences.")

# Load TTS model
tts = TTS("tts_models/multilingual/multi-dataset/your_tts").to("cpu")

# Generate audio
records = []
for i, text in enumerate(sentences):
    out_path = os.path.join(SYNTH_DIR, f"synthetic_{i}.wav")
    tts.tts_to_file(text=text, file_path=out_path)
    records.append({"path": out_path, "text": text.lower()})

# Save manifest
df = pd.DataFrame(records)
df.to_csv("synthetic_manifest.csv", index=False)

print("Synthetic audio generated and manifest saved.")
