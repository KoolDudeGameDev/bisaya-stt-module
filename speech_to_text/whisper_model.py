import whisper
import torch

# Load Whisper once
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisper.load_model("small", device=device)

def transcribe_with_whisper(audio_path):
    result = whisper_model.transcribe(audio_path, language="ceb", fp16=torch.cuda.is_available())
    return result['text']
