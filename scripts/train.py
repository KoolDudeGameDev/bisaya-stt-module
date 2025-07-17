from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, TrainingArguments, Trainer
from datasets import load_from_disk
import torch
import os
from datetime import datetime
from jiwer import wer
from train_callbacks import LiveSampleLogger

# === VERSION TAGS ===
DATASET_VERSION = "v1_training_ready"
PROCESSOR_VERSION = "v1_grapheme"
MODEL_VERSION = "v1_bisaya"

# === Load dataset and processor ===
raw_dataset = load_from_disk(f"data/processed/{DATASET_VERSION}")
processor = Wav2Vec2Processor.from_pretrained(f"processor/{PROCESSOR_VERSION}")

# === Filter out long samples (>15s) ===
MAX_INPUT_LENGTH_SEC = 15
max_len = int(processor.feature_extractor.sampling_rate * MAX_INPUT_LENGTH_SEC)
filtered_dataset = raw_dataset.filter(lambda x: len(x["input_values"]) <= max_len)

# === Split into train/test ===
dataset = filtered_dataset["train"].train_test_split(test_size=0.1)
print(f"✅ Dataset: {len(dataset['train'])} train / {len(dataset['test'])} test samples")

# === Load pre-trained model (Round 2 finetune) ===
model = Wav2Vec2ForCTC.from_pretrained(
    "kylegregory/wav2vec2-bisaya",  # From Round 1 checkpoint
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
)

# Manually resize the final layer (output projection) to match tokenizer vocab size
vocab_size = len(processor.tokenizer)
model.lm_head = torch.nn.Linear(model.lm_head.in_features, vocab_size, bias=True)
model.config.vocab_size = vocab_size

print(f"Tokenizer vocab size: {vocab_size}")
print(f"Model vocab size: {model.config.vocab_size}")

# === Data collator (with dynamic padding) ===
class DataCollatorCTCWithPadding:
    def __init__(self, processor, padding=True):
        self.processor = processor
        self.padding = padding

    def __call__(self, features):
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.feature_extractor.pad(input_features, padding=self.padding, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, padding=self.padding, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["input_ids"] == self.processor.tokenizer.pad_token_id, -100
        )

        batch["labels"] = labels
        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor)

# === Metric: Word Error Rate ===
def compute_metrics(pred):
    pred_ids = torch.argmax(torch.tensor(pred.predictions), dim=-1)
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    error_rate = wer(label_str, pred_str)

    # Log to file
    os.makedirs("docs", exist_ok=True)
    with open("docs/validation_metrics.md", "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat()} - WER: {error_rate:.4f}\n")

    return {"wer": error_rate}

# === Training arguments ===
training_args = TrainingArguments(
    output_dir=f"models/wav2vec2/{MODEL_VERSION}",
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=2,
    num_train_epochs=30,
    learning_rate=5e-5,
    lr_scheduler_type="linear",
    warmup_steps=500,
    logging_steps=10,
    logging_dir="./logs",
    logging_first_step=True,
    seed=42,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    remove_unused_columns=False,
    fp16=False,  # Set True if your GPU supports it (e.g. RTX 30xx)
)

# === Initialize trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=processor,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[LiveSampleLogger(processor, dataset["test"])],
)

# === Begin training ===
print("🚀 Starting Round 2 fine-tuning...")
#trainer.train()

# Resume Checkpoint training
# trainer.train(resume_from_checkpoint=True)
trainer.train(resume_from_checkpoint="models/wav2vec2/v1_bisaya/checkpoint-100")

# === Save model & processor ===
trainer.save_model(f"models/wav2vec2/{MODEL_VERSION}")
processor.save_pretrained(f"models/wav2vec2/{MODEL_VERSION}")

print(f"✅ Model saved to models/wav2vec2/{MODEL_VERSION}")
