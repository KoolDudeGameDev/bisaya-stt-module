# scripts/clean_manifest.py

import pandas as pd
import re
import argparse

def clean_text(text):
    # Normalize smart quotes to plain apostrophes (if needed)
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')

    # Optional: remove apostrophes altogether (common for grapheme tokenizers)
    text = text.replace("'", "")

    # Lowercase and allow only lowercase letters, space, and comma
    text = text.lower()
    text = re.sub(r"[^a-z ,]", "", text)

    return text

def main(input_path, output_path):
    df = pd.read_csv(input_path)

    if "text" not in df.columns:
        raise ValueError("CSV must contain a 'text' column.")

    df["text"] = df["text"].astype(str).apply(clean_text)
    df.to_csv(output_path, index=False)
    print(f"✅ Cleaned manifest saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output", type=str, required=True, help="Path to save cleaned CSV")

    args = parser.parse_args()
    main(args.input, args.output)
