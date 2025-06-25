import pandas as pd
from datasets import Dataset, Audio

# Load CSV
df = pd.read_csv("bisaya-dataset/bisaya_dataset.csv")

# Convert to Hugging Face Dataset format
ds = Dataset.from_pandas(df)

# Tell Hugging Face this column contains audio files
ds = ds.cast_column("path", Audio())

# Print first sample
print(ds[0])
