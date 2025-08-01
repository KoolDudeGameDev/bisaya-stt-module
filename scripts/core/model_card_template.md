---
language: ceb
tags:
  - bisaya
  - cebuano
  - wav2vec2
  - speech
  - asr
license: mit
datasets:
  - custom
metrics:
  - wer
---

# 📢 Wav2Vec2 Cebuano (Bisaya) Speech-to-Text Model

**Version:** `{{version_tag}}`  
**Last Trained:** `{{timestamp}}`

## 🔍 Overview

This model is a fine-tuned variant of [`facebook/wav2vec2-large-xlsr-53`](https://huggingface.co/facebook/wav2vec2-large-xlsr-53) for automatic speech recognition (ASR) on real and synthetic Cebuano (Bisaya) audio data.

It is designed to transcribe short utterances, spontaneous phrases, and conversational speech commonly found in marketplaces, bakeries, and informal daily interactions.

## 🗂️ Dataset

- **Corpus:** `{{dataset_version}}`
- **Source:** Real + synthetic audio samples  
- **Total Samples:** `{{num_samples}}`
- **Sampling Rate:** 16 kHz
- **Vocabulary Size:** `{{vocab_size}}`

## 📈 Evaluation

- **Metric:** Word Error Rate (WER)
- **Validation WER:** `{{wer}}`

The WER is tracked throughout training and logged in `logs/val_wer_history.csv`.

## 🛠️ Usage

```python
from transformers import pipeline

asr = pipeline("automatic-speech-recognition", model="kylegregory/wav2vec2-cebuano")
asr("path/to/audio.wav")
