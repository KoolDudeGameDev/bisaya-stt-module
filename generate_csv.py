import os
import csv

audio_dir = "bisaya-dataset/audio"
csv_path = "bisaya-dataset/bisaya_dataset.csv"

# Ensure the audio directory exists
if not os.path.exists(audio_dir):
    raise FileNotFoundError(f"Error: The directory {audio_dir} does not exist.")

# List all .wav files in the audio directory
files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]

# Check if any .wav files were found
if not files:
    print(f"Warning: No .wav files found in {audio_dir}")

# Create the CSV file and write the header
try:
    with open(csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["path", "text"])
        for f in files:
            writer.writerow([f"audio/{f}", ""])  # Placeholder: fill transcript later
except PermissionError:
    print(f"Error: No permission to write to {csv_path}")
