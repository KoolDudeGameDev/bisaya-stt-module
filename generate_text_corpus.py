# generate_text_corpus.py

import pandas as pd

# Try to load with proper encoding and quoting
try:
    df = pd.read_csv(
        "bisaya-dataset/DaddyBen.csv",
        encoding="cp1252",
        quoting=1,  # csv.QUOTE_ALL
        on_bad_lines="warn",  # skip problematic rows
        engine="python"       # more robust parser
    )
except Exception as e:
    print("Error reading CSV:", e)
    raise

# Show preview
print("Loaded data preview:")
print(df.head())

# Check columns
if "Line" not in df.columns:
    raise ValueError(f"Expected column 'Line', found: {list(df.columns)}")

# Drop NAs and deduplicate
lines = df["Line"].dropna().unique().tolist()
clean_lines = [line.strip() for line in lines if line.strip()]

# Save
with open("cebuano_text_corpus_extra.txt", "w", encoding="utf-8") as f:
    for line in clean_lines:
        f.write(line + "\n")

print(f"Saved {len(clean_lines)} lines to cebuano_text_corpus.txt")
