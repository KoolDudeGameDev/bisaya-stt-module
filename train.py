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


# Load dataset
dataset = load_from_disk("bisaya-training-ready-dataset")
processor = Wav2Vec2Processor.from_pretrained("processor")

# Split train/test
dataset = dataset.train_test_split(test_size=0.1)

# Model
model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-base-960h",
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
)

# Metrics


def compute_metrics(pred):
    pred_ids = torch.argmax(pred.predictions, dim=-1)
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    from jiwer import wer
    return {"wer": wer(label_str, pred_str)}


# Training args
training_args = TrainingArguments(
    output_dir="./wav2vec2-bisaya",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    # evaluation_strategy="steps",
    num_train_epochs=10,
    fp16=True,
    save_steps=100,
    # eval_steps=100,
    logging_steps=50,
    learning_rate=1e-4,
    save_total_limit=2,
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

trainer.train()
