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

# Wav2Vec2 Bisaya STT Model

**Version:** {{version_tag}}  
**Last Trained:** {{timestamp}}

## Overview

This is a fine-tuned [Wav2Vec2](https://huggingface.co/facebook/wav2vec2-large-xlsr-53) model for automatic speech recognition (ASR) on the Cebuano/Bisaya language.

## Dataset
- Corpus: `{{dataset_version}}`
- Number of audio samples: `{{num_samples}}`
- Vocab size: `{{vocab_size}}`

## Evaluation
- Word Error Rate (WER): `{{wer}}`

## Usage

```python
from transformers import pipeline

pipe = pipeline("automatic-speech-recognition", model="kylegregory/wav2vec2-bisaya")
pipe("path/to/audio.wav")
