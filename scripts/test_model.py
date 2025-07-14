from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

model = Wav2Vec2ForCTC.from_pretrained("./wav2vec2-bisaya")
processor = Wav2Vec2Processor.from_pretrained("./wav2vec2-bisaya")

print("Model and processor loaded successfully.")
