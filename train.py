from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, TrainingArguments, Trainer
from datasets import load_from_disk
import torch


class DataCollatorCTCWithPadding:
    def __init__(self, processor, padding=True):
        self.processor = processor
        self.padding = padding

    def __call__(self, features):
        input_features = [{"input_values": f["input_values"]}
                          for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            return_tensors="pt",
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["input_ids"] == self.processor.tokenizer.pad_token_id, -100
        )

        batch["labels"] = labels

        return batch


# Load preprocessed Bisaya dataset from disk
dataset = load_from_disk("bisaya-training-ready-dataset")
processor = Wav2Vec2Processor.from_pretrained("processor")

# Filter out examples longer than 15 seconds
MAX_INPUT_LENGTH_SEC = 15

dataset = dataset.filter(
    lambda example: len(example["input_values"]) <= int(
        processor.feature_extractor.sampling_rate * MAX_INPUT_LENGTH_SEC
    )
)

# Split into 90% train and 10% test
dataset = dataset.train_test_split(test_size=0.1)

# Load pre-trained Wav2Vec2 model for Bisaya language
model = Wav2Vec2ForCTC.from_pretrained(
    "kylegregory/wav2vec2-bisaya",
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
)

# Metrics function to compute WER and log to file
def compute_metrics(pred):
    import os
    from datetime import datetime
    from jiwer import wer

    # Decode predictions
    pred_ids = torch.argmax(torch.from_numpy(pred.predictions), dim=-1)
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    # Compute WER
    error_rate = wer(label_str, pred_str)

    # Prepare log line
    timestamp = datetime.now().isoformat()
    log_line = f"{timestamp} - WER: {error_rate:.4f}\n"

    # Ensure docs directory exists
    os.makedirs("docs", exist_ok=True)

    # Append to metrics log file
    metrics_path = os.path.join("docs", "validation_metrics.md")
    with open(metrics_path, "a", encoding="utf-8") as f:
        f.write(log_line)

    # Also return metric dict for Trainer
    return {"wer": error_rate}



# Configure training arguments with optimized settings
training_args = TrainingArguments(
    output_dir="./wav2vec2-bisaya",
    save_strategy="steps",
    eval_strategy="steps",
    eval_steps=100,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=2,
    num_train_epochs=30,
    fp16=False, # Disable FP16 for better compatibility
    save_steps=100,
    logging_steps=10,
    learning_rate=5e-5,
    save_total_limit=3,
    lr_scheduler_type="linear",
    warmup_steps=500,
    seed=42,
    load_best_model_at_end=True,  # Automatically load best model based on eval metric
    metric_for_best_model="wer",  # Optimize for lowest WER
    greater_is_better=False,      # Lower WER is better
    logging_dir="./logs",         # Centralized logging
    remove_unused_columns=False,  # Preserve all columns for debugging
)

# Data collator using DataCollatorWithPadding
data_collator = DataCollatorCTCWithPadding(
    processor=processor,
    padding=True
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=processor,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Start training
print("Starting training...")
trainer.train()
#trainer.train(resume_from_checkpoint=True)


# Save model and processor
trainer.save_model("./wav2vec2-bisaya")
processor.save_pretrained("./wav2vec2-bisaya")


