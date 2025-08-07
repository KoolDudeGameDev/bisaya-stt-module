import pandas as pd
from datasets import load_from_disk
import os

# === CONFIGURATION ===
CLEANED_MANIFEST = "data/final/cleaned_train_manifest.csv"
PROCESSED_DATASET = "data/processed/v1_training_ready_grapheme"

def verify_manifest():
    print("📊 Verifying cleaned manifest CSV...")
    try:
        df = pd.read_csv(CLEANED_MANIFEST)
        print(f"✅ Rows in cleaned manifest: {len(df)}")
        print(f"🧾 Sample entries:\n{df.head()}")

        # Verify audio file existence
        missing_audio = df[~df['path'].apply(os.path.exists)]
        if not missing_audio.empty:
            print(f"⚠️ Missing audio files: {len(missing_audio)}")
            print(missing_audio[['path', 'text']])
        else:
            print("✅ All audio file paths exist.")
        return df
    except Exception as e:
        print("❌ Error reading manifest:", str(e))
        raise

def verify_dataset(manifest_df):
    print("\n📦 Loading Hugging Face dataset...")
    try:
        dataset_dict = load_from_disk(PROCESSED_DATASET)
        if "train" not in dataset_dict:
            raise ValueError("❌ 'train' split not found in the processed dataset.")
        
        dataset = dataset_dict["train"]
        print(f"✅ Total training samples: {len(dataset)}")

        # Convert to DataFrame for merging with manifest
        dataset_df = dataset.to_pandas()

        # Reattach original audio path from manifest
        if len(dataset_df) != len(manifest_df):
            raise ValueError("❌ Mismatch between dataset and manifest lengths.")
        
        dataset_df["audio_path"] = manifest_df["path"].values

        # Show sample entries
        print("\n🔍 Sample entries from dataset:")
        for i in range(min(3, len(dataset_df))):
            row = dataset_df.iloc[i]
            print(f"\n[{i}]")
            print("  📝 Text:", row["text"])
            print("  🎧 Audio Path:", row["audio_path"])
            print("  🔢 Input Values:", len(row["input_values"]) if isinstance(row["input_values"], list) else "[Invalid]")
            print("  🔡 Labels ({}): {}".format(
                len(row["labels"]) if isinstance(row["labels"], list) else 0,
                row["labels"][:20] if isinstance(row["labels"], list) else "[Invalid]"
            ))

        return dataset
    except Exception as e:
        print("❌ Error loading dataset:", str(e))
        raise

def main():
    manifest_df = verify_manifest()
    verify_dataset(manifest_df)

if __name__ == "__main__":
    main()
