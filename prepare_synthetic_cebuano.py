from TTS.api import TTS
import os
import pandas as pd
import random
import re
from glob import glob

# Output directory
SYNTH_DIR = "synthetic_bisaya_audio"
os.makedirs(SYNTH_DIR, exist_ok=True)

# Input text corpus file
TEXT_CORPUS = "cebuano_text_corpus_extra.txt"

# Determine the starting index based on existing audio files
existing_wavs = glob(os.path.join(SYNTH_DIR, "synthetic_*.wav"))
existing_indices = [
    int(re.search(r"synthetic_(\d+)\.wav", f).group(1)) for f in existing_wavs if re.search(r"synthetic_(\d+)\.wav", f)
]
start_idx = max(existing_indices) + 1 if existing_indices else 0

# Load lines and filter invalid ones
with open(TEXT_CORPUS, "r", encoding="utf-8") as f:
    raw_sentences = [line.strip() for line in f if line.strip()]

# Filter out junk lines (e.g., numbers only, symbols)
sentences = [
    s for s in raw_sentences if len(s) >= 3 and re.search(r'[a-zA-Z]', s)
]
skipped_sentences = [
    s for s in raw_sentences if s not in sentences
]

# Log skipped lines
if skipped_sentences:
    with open("skipped_synthetic_lines.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(skipped_sentences))
    print(f"⚠️ Skipped {len(skipped_sentences)} malformed/invalid lines (logged to skipped_synthetic_lines.txt)")

print(f"✅ {len(sentences)} valid sentences loaded from '{TEXT_CORPUS}'.")

# Load TTS model
tts = TTS("tts_models/multilingual/multi-dataset/your_tts").to("cpu")

# Clean speaker names
available_speakers = sorted(list(set([s.strip() for s in tts.speakers])))
selected_language = "en"  # closest match since Cebuano is not natively supported

print(f"🎙 Available speakers: {available_speakers}")

# Generate audio
records = []
for i, text in enumerate(sentences):
    out_index = start_idx + i
    out_path = os.path.join(SYNTH_DIR, f"synthetic_{out_index}.wav")
    random_speaker = random.choice(available_speakers)
    print(f"🎧 [{out_index}] Generating audio: '{text}' | Speaker: {random_speaker}")

    try:
        tts.tts_to_file(
            text=text,
            file_path=out_path,
            speaker=random_speaker,
            language=selected_language
        )
        records.append({
            "path": out_path,
            "text": text.lower(),
            "speaker": random_speaker
        })
    except Exception as e:
        print(f"❌ Failed to synthesize line: '{text}' | Error: {e}")
        with open("failed_synthetic_lines.txt", "a", encoding="utf-8") as log_f:
            log_f.write(f"{text}\n")

# Save manifest
if records:
    df = pd.DataFrame(records)
    if os.path.exists("synthetic_manifest.csv"):
        df_existing = pd.read_csv("synthetic_manifest.csv")
        df = pd.concat([df_existing, df], ignore_index=True)
    df.to_csv("synthetic_manifest.csv", index=False)
    print(f"✅ Appended {len(records)} entries to synthetic_manifest.csv")
else:
    print("⚠️ No audio generated.")

print("🎯 Synthetic TTS generation complete.")
