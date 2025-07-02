from transformers import pipeline
import torch
import librosa
import soundfile as sf
import os

print("[INFO] Loaded ASR model: kylegregory/wav2vec2-bisaya")

# Hugging Face ASR pipeline abstraction — Wav2Vec2 model
asr_pipeline = pipeline(
    task="automatic-speech-recognition",
    chunk_length_s=15,
    model="kylegregory/wav2vec2-bisaya",
    device=0 if torch.cuda.is_available() else -1
)


def transcribe_with_wav2vec2(file_path):
    # Load and resample the audio file
    speech, sample_rate = librosa.load(file_path, sr=16000)

    # Save the resampled audio to a temporary WAV file
    temp_wav = file_path.replace('.wav', '_16k.wav')
    sf.write(temp_wav, speech, 16000)

    # Transcribe using the pipeline
    try:
        result = asr_pipeline(temp_wav)
    except Exception as e:
        raise RuntimeError(f"Transcription failed: {e}")
    finally:
        # Ensure the temporary file is removed even if an error occurs
        if os.path.exists(temp_wav):
            os.remove(temp_wav)

    return result['text']


def is_transcription_confident(text: str) -> bool:
    if not text:
        return False
    return len(text.split()) >= 3


