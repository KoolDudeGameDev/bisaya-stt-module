from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, TrainingArguments, Trainer
from datasets import load_from_disk
import torch
import os
from datetime import datetime
from jiwer import wer
from train_callbacks import LiveSampleLogger
from loss_plot_callback import LossPlotCallback

torch.set_num_threads(os.cpu_count())

# === VERSION TAGS ===
DATASET_VERSION = "v1_training_ready_grapheme"
PROCESSOR_VERSION = "v1_grapheme"
MODEL_VERSION = "v1_bisaya"

# === Load dataset and processor ===
print("🔍 Loading dataset and processor...")
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
print("🔧 Loading base model...")
model = Wav2Vec2ForCTC.from_pretrained(
    "kylegregory/wav2vec2-bisaya", 
    ctc_loss_reduction="mean",
    ctc_zero_infinity=True,
    pad_token_id=processor.tokenizer.pad_token_id,
)

# === Resize model output to match new tokenizer vocab ===
vocab_size = len(processor.tokenizer)
model.lm_head = torch.nn.Linear(model.lm_head.in_features, vocab_size, bias=True)
model.config.vocab_size = vocab_size
print(f"🧠 Tokenizer vocab size: {vocab_size}")
print(f"🧠 Model vocab size: {model.config.vocab_size}")

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

    print("\n🔍 Live Eval Debug:")
    for ref, hyp in list(zip(label_str, pred_str))[:3]:
        print("🧾 REF:", ref)
        print("🔊 HYP:", hyp)

    error_rate = wer(label_str, pred_str)

    os.makedirs("docs", exist_ok=True)
    with open("docs/validation_metrics.md", "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat()} - WER: {error_rate:.4f}\n")

    return {"wer": error_rate}

# === Training Arguments ===
training_args = TrainingArguments(
    output_dir=f"models/wav2vec2/{MODEL_VERSION}",
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    num_train_epochs=30,
    learning_rate=5e-5,
    lr_scheduler_type="linear",
    warmup_steps=500,
    logging_steps=50,
    logging_dir="./logs",
    logging_first_step=True,
    seed=42,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    remove_unused_columns=False,
    max_grad_norm=1.0,
    fp16=False,
)

# === Freeze feature extractor (optional but useful) ===
model.freeze_feature_encoder()

# === Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=processor,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[LiveSampleLogger(processor, dataset["test"]), LossPlotCallback()],
)

# === Begin training ===
print("🚀 Starting Round 2 fine-tuning...")
trainer.train()

# === Save model & processor ===
trainer.save_model(f"models/wav2vec2/{MODEL_VERSION}")
processor.save_pretrained(f"models/wav2vec2/{MODEL_VERSION}")
print(f"✅ Model saved to models/wav2vec2/{MODEL_VERSION}")

# === Push to Hugging Face ===
print("📤 Uploading best checkpoint to Hugging Face...")
from scripts.auto_push_checkpoint import push_checkpoint
push_checkpoint(model_dir=f"models/wav2vec2/{MODEL_VERSION}", tag=MODEL_VERSION)
print("🎉 Upload complete.")
