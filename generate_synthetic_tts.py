# generate_synthetic_tts.py

from TTS.api import TTS
import os
import pandas as pd

# Define output dir
SYNTH_DIR = "synthetic_bisaya_audio"
os.makedirs(SYNTH_DIR, exist_ok=True)

# Example sentences to generate
sentences = [
    "Kumusta ka?",
    "Palihug isulat ang imong ngalan.",
    "Aduna ba kay pangutana?",
    "Pila ang presyo niini?",
    "Salamat kaayo."
]

# Load a multilingual TTS model
tts = TTS("tts_models/multilingual/multi-dataset/your_tts").to("cpu")

# Generate audio
records = []
for i, text in enumerate(sentences):
    out_path = os.path.join(SYNTH_DIR, f"synthetic_{i}.wav")
    tts.tts_to_file(text=text, file_path=out_path, speaker_wav=None)
    records.append({"path": out_path, "text": text.lower()})

# Save manifest
df = pd.DataFrame(records)
df.to_csv("synthetic_manifest.csv", index=False)

print("Synthetic audio generated.")
