from TTS.api import TTS
import os
import pandas as pd

# Output dir
SYNTH_DIR = "synthetic_bisaya_audio"
os.makedirs(SYNTH_DIR, exist_ok=True)

# Load lines from text corpus
with open("cebuano_text_corpus.txt", "r", encoding="utf-8") as f:
    sentences = [line.strip() for line in f if line.strip()]

print(f"Loaded {len(sentences)} sentences.")

# Load multilingual multi-speaker model
tts = TTS("tts_models/multilingual/multi-dataset/your_tts").to("cpu")

# Sanitize speaker list
available_speakers = list(set([s.strip() for s in tts.speakers]))
print("Available speakers:", available_speakers)

# Choose preferred speaker and language manually
selected_speaker = "female-en-5"
selected_language = "en"  # fallback, since Cebuano is unsupported

# Generate audio
records = []
for i, text in enumerate(sentences):
    out_path = os.path.join(SYNTH_DIR, f"synthetic_{i}.wav")
    print(f"Generating audio for: {text} with speaker: {selected_speaker}")
    tts.tts_to_file(
        text=text,
        file_path=out_path,
        speaker=selected_speaker,
        language=selected_language
    )
    records.append({"path": out_path, "text": text.lower()})

# Save manifest
df = pd.DataFrame(records)
df.to_csv("synthetic_manifest.csv", index=False)

print(f"✅ Generated {len(records)} synthetic audio files.")
