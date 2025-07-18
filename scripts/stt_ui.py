import streamlit as st
import torchaudio
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
import av
import numpy as np
import tempfile
import os

MODEL_ID = "kylegregory/wav2vec2-bisaya"

# Load model and processor with caching
@st.cache_resource
def load_model():
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
    return processor, model

# Custom audio processor class for mic input
class AudioProcessor:
    def __init__(self):
        self.frames = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        pcm = frame.to_ndarray().flatten()
        self.frames.append(pcm)
        return frame

    def get_audio(self):
        audio = np.concatenate(self.frames).astype(np.float32)
        return torch.from_numpy(audio).unsqueeze(0)

# App UI
st.title("🗣️ Bisaya Speech-to-Text")
st.markdown("Record your voice or upload a `.wav` file (mono, 16kHz) to transcribe it.")

# Load model
processor, model = load_model()

# --- Microphone Recording Section ---
st.header("🎙️ Record from Microphone")
audio_processor = AudioProcessor()

webrtc_ctx = webrtc_streamer(
    key="stt",
    mode=WebRtcMode.SENDONLY,
    in_audio_enabled=True,
    client_settings=ClientSettings(media_stream_constraints={"audio": True, "video": False}),
    audio_receiver_size=1024,
    sendback_audio=False,
    audio_processor_factory=lambda: audio_processor,
)

if st.button("Transcribe Recording") and audio_processor.frames:
    st.info("🔄 Processing audio from microphone...")
    waveform = audio_processor.get_audio()

    # Ensure correct sample rate
    waveform = torchaudio.transforms.Resample(orig_freq=48000, new_freq=16000)(waveform)

    with torch.no_grad():
        inputs = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt").input_values
        logits = model(inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]

    st.success("📝 Transcription:")
    st.markdown(f"**{transcription}**")

# --- File Upload Fallback Section ---
st.header("📂 Or Upload a WAV File")
audio_file = st.file_uploader("Choose a `.wav` file", type=["wav"])

if audio_file:
    st.audio(audio_file)
    st.info("🔄 Processing uploaded audio...")
    waveform, sr = torchaudio.load(audio_file)

    if sr != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)

    with torch.no_grad():
        inputs = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt").input_values
        logits = model(inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]

    st.success("📝 Transcription:")
    st.markdown(f"**{transcription}**")
