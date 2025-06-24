from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import soundfile as sf
import io


class STTModel:
    """
    Speech-to-Text model using Wav2Vec2 for CTC-based transcription.
    """

    def __init__(self, model_path):
        """
        Initialize the STTModel with a given pretrained model path.

        Args:
            model_path (str): Path to the pretrained Wav2Vec2 model.
        """
        self.processor = Wav2Vec2Processor.from_pretrained(model_path)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_path)
        self.model.eval()

    def transcribe(self, audio_bytes):
        """
        Transcribe audio bytes into text.

        Args:
            audio_bytes (bytes): Audio data in bytes.

        Returns:
            str: Transcribed text from the audio.
        """
        audio, _ = sf.read(io.BytesIO(audio_bytes))
        inputs = self.processor(
            audio, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]
            return transcription
