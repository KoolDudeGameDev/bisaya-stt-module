from datasets import load_from_disk

# Load the preprocessed dataset
dataset = load_from_disk("bisaya-preprocessed-dataset")

print(dataset.features)

# Display the number of entries
print(f"Total entries: {len(dataset)}")

# Display one sample
print(dataset[0])

# Optional: iterate through all to verify audio loading
for i in range(3):  # check first 3 examples
    audio = dataset[i]["path"]
    text = dataset[i]["text"]

    # Conditionally get 'category' if present
    if "category" in dataset.features:
        category = dataset[i]["category"]
    else:
        category = "N/A"

    print(f"\nSample {i + 1}:")
    print(f"  Audio Path: {audio['path']}")
    print(f"  Sampling Rate: {audio['sampling_rate']}")
    print(f"  Waveform Length: {len(audio['array'])}")
    print(f"  Text: {text}")
    print(f"  Category: {category}")
