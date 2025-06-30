from datasets import load_from_disk
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, TrainingArguments, Trainer, DataCollatorCTCWithPadding
import torch

# Load
dataset = load_from_disk("bisaya-training-ready-dataset")
processor = Wav2Vec2Processor.from_pretrained("processor/")

# Split train/test
dataset = dataset.train_test_split(test_size=0.1)

model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-base-960h",
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
)


def compute_metrics(pred):
    pred_ids = torch.argmax(pred.predictions, dim=-1)
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    from jiwer import wer
    return {"wer": wer(label_str, pred_str)}


training_args = TrainingArguments(
    output_dir="./wav2vec2-bisaya",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    evaluation_strategy="steps",
    num_train_epochs=10,
    fp16=True,
    save_steps=100,
    eval_steps=100,
    logging_steps=50,
    learning_rate=1e-4,
    save_total_limit=2,
)

data_collator = DataCollatorCTCWithPadding(processor=processor)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=processor.feature_extractor,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
