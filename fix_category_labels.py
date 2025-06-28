import pandas as pd

csv_path = "bisaya-dataset/bisaya_dataset.csv"

df = pd.read_csv(csv_path)

# Force everything to string BEFORE replacing
df["category"] = df["category"].astype(str)

# Apply replacements on string
df["category"] = df["category"].replace({
    "1": "number",
    "2": "bread"
})

# To be 100% sure, re-assert that it's object dtype
df["category"] = df["category"].astype(str)

df.to_csv(csv_path, index=False)

print("[✅] Categories updated successfully.")
