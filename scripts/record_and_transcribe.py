import os
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import pandas as pd

# Config
AUDIO_DIR = "bisaya-dataset/audio"
CSV_PATH = "bisaya-dataset/bisaya_dataset.csv"
SAMPLE_RATE = 16000
CHANNELS = 1
DURATION = 5  # default duration in seconds

# Ensure folders exist
os.makedirs(AUDIO_DIR, exist_ok=True)

# Initialize or load dataset
if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
else:
    df = pd.DataFrame(columns=["path", "text", "category"])

def record_sample(index):
    filename = f"sample{index}.wav"
    path = os.path.join(AUDIO_DIR, filename)

    print(f"\n[🎙️] Recording sample {index} for {DURATION} seconds...")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16')
    sd.wait()
    wav.write(path, SAMPLE_RATE, audio)
    print(f"[✅] Saved to {path}")

    # Playback for review
    print("[🔊] Playing back the recording...")
    sd.play(audio, SAMPLE_RATE)
    sd.wait()

    # Confirm quality
    review = input("Keep this recording? (y/n): ").strip().lower()
    if review != 'y':
        os.remove(path)
        print("[🗑️] Recording deleted. Please re-record.")
        return False

    transcript = input("Enter transcription (Bisaya): ").strip()
    category = input("Enter category (e.g., number, bread, greeting): ").strip().lower()

    df.loc[len(df.index)] = [f"audio/{filename}", transcript, category]
    df.to_csv(CSV_PATH, index=False)
    print(f"[📝] Entry saved to {CSV_PATH}")
    return True

# Main loop
if __name__ == "__main__":
    sample_index = len(df) + 1
    while True:
        success = record_sample(sample_index)
        if success:
            sample_index += 1
        cont = input("Record another? (y/n): ").strip().lower()
        if cont != 'y':
            break
