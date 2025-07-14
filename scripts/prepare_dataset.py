import os
import pandas as pd
from datasets import Dataset, Audio, load_dataset, concatenate_datasets
import soundfile as sf

CSV_PATH = "bisaya-dataset/bisaya_dataset_clean.csv"

def validate_and_load():
    df = pd.read_csv(CSV_PATH)

    if not {"path", "text"}.issubset(df.columns):
        raise ValueError("CSV must have columns: path, text")

    # Convert relative paths to absolute
    df["path"] = df["path"].apply(lambda x: os.path.join("bisaya-dataset", x))
    return df


def resample_wav(input_path, target_sr=16000):
    data, samplerate = sf.read(input_path)
    if samplerate != target_sr:
        import librosa
        data = librosa.resample(
            data.T, orig_sr=samplerate, target_sr=target_sr)
        data = data.T
        sf.write(input_path, data, target_sr)
        print(f"[🔄] Resampled {input_path} to {target_sr}Hz")


def main():
    df = validate_and_load()
    print(f"[✅] Loaded {len(df)} entries.")

    # Resample all audio files to 16kHz
    for p in df["path"]:
        resample_wav(p, target_sr=16000)

    # Create Hugging Face Dataset
    dataset = Dataset.from_pandas(df)
    dataset = dataset.cast_column("path", Audio(sampling_rate=16000))

    # Save to disk
    dataset.save_to_disk("bisaya-preprocessed-dataset")
    print("[💾] Dataset saved to 'bisaya-preprocessed-dataset'")


if __name__ == "__main__":
    main()

# This script prepares the Bisaya dataset for use with Hugging Face's datasets library.
