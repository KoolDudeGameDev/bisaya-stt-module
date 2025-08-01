# scripts/batch_infer_folder.py

import os
import pandas as pd
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

AUDIO_DIR = "samples/"
MODEL_DIR = "models/wav2vec2/v1_cebuano"
OUTPUT_CSV = "transcriptions.csv"

processor = Wav2Vec2Processor.from_pretrained(MODEL_DIR)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_DIR)
model.eval()

results = []

for filename in os.listdir(AUDIO_DIR):
    if filename.endswith(".wav"):
        path = os.path.join(AUDIO_DIR, filename)
        waveform, sample_rate = torchaudio.load(path)

        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)

        input_values = processor(waveform.squeeze().numpy(), return_tensors="pt", sampling_rate=16000).input_values

        with torch.no_grad():
            logits = model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)[0]

        results.append({"filename": filename, "transcription": transcription})

pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
print(f"✅ Saved all transcriptions to: {OUTPUT_CSV}")
