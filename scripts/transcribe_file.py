import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

MODEL_ID = "kylegregory/wav2vec2-bisaya"

def transcribe(audio_path: str):
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)

    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        waveform = resampler(waveform)

    input_values = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt").input_values
    with torch.no_grad():
        logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]

    return transcription

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python scripts/transcribe_file.py path/to/audio.wav")
        sys.exit(1)

    print(f"[🎧] Transcribing {sys.argv[1]}...")
    result = transcribe(sys.argv[1])
    print(f"[📝] Transcript: {result}")
