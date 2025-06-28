import pandas as pd

csv_path = "bisaya-dataset/bisaya_dataset.csv"

df = pd.read_csv(csv_path)

print("[INFO] Column types:")
print(df.dtypes)

print("\n[INFO] Unique category values:")
print(df['category'].unique())
