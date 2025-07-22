import streamlit as st
import torchaudio
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
import av
import numpy as np
import inspect

MODEL_ID = "kylegregory/wav2vec2-bisaya"

# === Load Model & Processor ===
@st.cache_resource
def load_model():
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
    return processor, model

# === Custom Audio Collector ===
class AudioProcessor:
    def __init__(self):
        self.frames = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        pcm = frame.to_ndarray().flatten()
        self.frames.append(pcm)
        return frame

    def get_audio(self):
        return torch.tensor(np.concatenate(self.frames), dtype=torch.float32).unsqueeze(0)

# === STT UI Logic ===
def record_and_transcribe():
    st.header("🎙️ Record from Microphone")
    processor, model = load_model()
    audio_processor = AudioProcessor()

    # Detect if 'in_audio_enabled' is a valid parameter for webrtc_streamer
    webrtc_args = {
        "key": "stt",
        "mode": WebRtcMode.SENDONLY,
        "client_settings": ClientSettings(media_stream_constraints={"audio": True, "video": False}),
        "audio_receiver_size": 1024,
        "sendback_audio": False,
        "audio_processor_factory": lambda: audio_processor,
    }

    if "in_audio_enabled" in inspect.signature(webrtc_streamer).parameters:
        webrtc_args["in_audio_enabled"] = True

    webrtc_streamer(**webrtc_args)

    if st.button("🔎 Transcribe Recording") and audio_processor.frames:
        st.info("⏳ Processing microphone input...")
        waveform = audio_processor.get_audio()
        waveform = torchaudio.transforms.Resample(orig_freq=48000, new_freq=16000)(waveform)

        with torch.no_grad():
            inputs = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt").input_values
            logits = model(inputs).logits
            pred_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(pred_ids)[0]

        st.success("📝 Transcription from Microphone:")
        st.markdown(f"`{transcription.strip()}`")

    # === File Upload Section ===
    st.markdown("---")
    st.header("📂 Or Upload a WAV File")
    audio_file = st.file_uploader("Upload mono 16kHz `.wav` file", type=["wav"])

    if audio_file:
        st.audio(audio_file)
        waveform, sr = torchaudio.load(audio_file)

        if sr != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)

        with torch.no_grad():
            inputs = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt").input_values
            logits = model(inputs).logits
            pred_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(pred_ids)[0]

        st.success("📝 Transcription from Uploaded File:")
        st.markdown(f"`{transcription.strip()}`")
