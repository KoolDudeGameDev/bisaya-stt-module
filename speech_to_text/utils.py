from transformers import pipeline
import torch
import librosa
import soundfile as sf
import os

# Hugging Face ASR pipeline abstraction — Wav2Vec2 model
asr_pipeline = pipeline(
    task="automatic-speech-recognition",
    # Replace with fine-tuned Bisaya model in the future
    model="facebook/wav2vec2-base-960h",
    device=0 if torch.cuda.is_available() else -1
)


def transcribe_audio_file(file_path):
    # Load and resample the audio file
    speech, sample_rate = librosa.load(file_path, sr=16000)

    # Save the resampled audio to a temporary WAV file
    temp_wav = file_path.replace('.wav', '_16k.wav')
    sf.write(temp_wav, speech, 16000)

    # Transcribe using the pipeline
    result = asr_pipeline(temp_wav)

    # Clean up the temporary file
    os.remove(temp_wav)

    return result['text']
