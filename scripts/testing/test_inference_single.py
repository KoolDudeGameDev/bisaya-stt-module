# scripts/test_inference_single.py

import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torchaudio

MODEL_DIR = "models/wav2vec2-cebuano"
AUDIO_PATH = "data/raw/audio/real_v1/real_v1_kyle_0363.wav"

print("🔁 Loading model and processor...")
processor = Wav2Vec2Processor.from_pretrained(MODEL_DIR)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_DIR)
model.eval()

print("🔊 Loading audio...")
waveform, sample_rate = torchaudio.load(AUDIO_PATH)
if sample_rate != processor.feature_extractor.sampling_rate:
    waveform = torchaudio.functional.resample(waveform, sample_rate, processor.feature_extractor.sampling_rate)

input_values = processor(waveform.squeeze().numpy(), return_tensors="pt", sampling_rate=16000).input_values

print("🧠 Performing inference...")
with torch.no_grad():
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)

transcription = processor.batch_decode(predicted_ids)[0]
print(f"📜 Transcription: {transcription}")
