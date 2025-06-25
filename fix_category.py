import pandas as pd

csv_path = "bisaya-dataset/bisaya_dataset.csv"

df = pd.read_csv(csv_path)
df['category'] = df['category'].replace("2", "bread")
df.to_csv(csv_path, index=False)

print("[✅] Replaced category '2' with 'bread' in bisaya_dataset.csv")
