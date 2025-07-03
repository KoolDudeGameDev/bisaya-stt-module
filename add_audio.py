import os
import pandas as pd
import sounddevice as sd
import soundfile as sf

# Config
AUDIO_DIR = "bisaya-dataset/audio"
CSV_PATH = "bisaya-dataset/bisaya_dataset.csv"

# Ensure folders exist
os.makedirs(AUDIO_DIR, exist_ok=True)

# Load dataset
if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
else:
    df = pd.DataFrame(columns=["path", "text", "category"])

def add_sample():
    filename = input("Enter the WAV filename (copy to audio folder first): ").strip()
    path = os.path.join(AUDIO_DIR, filename)

    if not os.path.exists(path):
        print("[❌] File does not exist. Please check the path.")
        return

    # Playback for review
    #print("[🔊] Playing audio for review...")
    #data, samplerate = sf.read(path)
    #sd.play(data, samplerate)
    #sd.wait()

    review = input("Use this recording? (y/n): ").strip().lower()
    if review != 'y':
        print("[⏭️] Skipping this file.")
        return

    transcript = input("Enter transcription (Bisaya): ").strip()
    category = input("Enter category (or leave blank): ").strip().lower()
    
    df.loc[len(df.index)] = [f"audio/{filename}", transcript, category]
    df.to_csv(CSV_PATH, index=False)
    print(f"[✅] Entry saved to {CSV_PATH}")

# Main loop
if __name__ == "__main__":
    while True:
        add_sample()
        cont = input("Add another file? (y/n): ").strip().lower()
        if cont != 'y':
            break
