import streamlit as st
import torchaudio
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

MODEL_ID = "kylegregory/wav2vec2-bisaya"

@st.cache_resource
def load_model():
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
    return processor, model

st.title("🗣️ Bisaya Speech-to-Text")
st.markdown("Upload a `.wav` audio file (mono, 16kHz) to transcribe it.")

audio_file = st.file_uploader("Choose a WAV file", type=["wav"])

if audio_file:
    processor, model = load_model()
    waveform, sr = torchaudio.load(audio_file)

    if sr != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)

    with torch.no_grad():
        input_values = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt").input_values
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]

    st.audio(audio_file)
    st.markdown("### 📝 Transcription")
    st.success(transcription)
