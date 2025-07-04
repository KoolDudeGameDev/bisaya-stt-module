# prepare_common_voice.py

from datasets import load_dataset, Dataset, concatenate_datasets
import torchaudio
import os
import pandas as pd

# Output directory for processed audio
OUTPUT_DIR = "common_voice_cebuano_audio"

# Ensure output dir exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading Common Voice Cebuano dataset...")
dataset = load_dataset(
    "mozilla-foundation/common_voice_17_0",
    "ceb",
    split="train+validation+test"
)

print(f"Dataset loaded with {len(dataset)} entries.")

# Function to preprocess each sample


def preprocess_and_save(batch, idx):
    # Load audio
    speech_array, sampling_rate = torchaudio.load(batch["path"])

    # Resample to 16kHz if necessary
    if sampling_rate != 16000:
        resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
        speech_array = resampler(speech_array)

    # Normalize (optional)
    speech_array = speech_array / speech_array.abs().max()

    # Create output path
    output_path = os.path.join(OUTPUT_DIR, f"cv_cebuano_{idx}.wav")

    # Save as WAV
    torchaudio.save(output_path, speech_array, 16000)

    # Return dict with processed info
    return {
        "path": output_path,
        "text": batch["sentence"].strip().lower()
    }


# Process all samples
processed = []
print("Preprocessing and saving audio files...")
for idx, sample in enumerate(dataset):
    processed.append(preprocess_and_save(sample, idx))

# Convert to DataFrame
df = pd.DataFrame(processed)

# Save manifest
df.to_csv("common_voice_cebuano_manifest.csv", index=False)
print("Manifest saved as 'common_voice_cebuano_manifest.csv'")
