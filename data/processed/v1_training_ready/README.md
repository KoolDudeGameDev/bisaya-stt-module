---
datasets:
  - name: bisaya-stt
  - language: ceb
  - license: mit
  - task_ids: automatic-speech-recognition
pretty_name: Bisaya Speech Recognition Dataset (v1)
---

# 📚 Bisaya Speech Recognition Dataset (v1)

This dataset is used for training an **automatic speech recognition (ASR)** model for the Cebuano (Bisaya) language. It is composed of:

- Real recorded Bisaya speech
- Synthetic audio generated via [MMS TTS](https://huggingface.co/facebook/mms-tts-ceb)
- Cleaned transcriptions tokenized at the grapheme level

---

## 💾 Dataset Structure

Each sample contains:

- `path`: path to the WAV audio file (16kHz, mono)
- `text`: ground-truth transcription in Bisaya
- `source`: data source (e.g., `synthetic_v1`, `manual`, etc.)
- `duration_sec`: duration of the audio clip
- `filename`: original filename of the audio

---

## 📐 Statistics

- 📄 Total samples: 3,423
- 🗣️ Language: Cebuano (Bisaya)
- 🔊 Sampling rate: 16kHz
- 🏗️ Tokenizer: Grapheme-level tokenizer with CTC-compatible symbols

---

## 🏗️ Usage

This dataset was processed using `datasets.DatasetDict` and saved via `.save_to_disk(...)`.

You can load it with:

```python
from datasets import load_from_disk
dataset = load_from_disk("data/processed/v1_training_ready")
