from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, TrainingArguments, Trainer
from datasets import load_from_disk
import torch, os, shutil
from datetime import datetime
from jiwer import wer
from train_callbacks import LiveSampleLogger
from loss_plot_callback import LossPlotCallback

torch.set_num_threads(os.cpu_count())

# === Config ===
DATASET_VERSION = "v1_training_ready_grapheme"
PROCESSOR_VERSION = "v1_grapheme"
MODEL_VERSION = "v1_bisaya"

# === Load Dataset ===
print("🔍 Loading dataset and processor...")
raw_dataset = load_from_disk(f"data/processed/{DATASET_VERSION}")
processor = Wav2Vec2Processor.from_pretrained(f"processor/{PROCESSOR_VERSION}")

MAX_INPUT_LENGTH_SEC = 15
max_len = int(processor.feature_extractor.sampling_rate * MAX_INPUT_LENGTH_SEC)
filtered_dataset = raw_dataset.filter(lambda x: len(x["input_values"]) <= max_len)

dataset = filtered_dataset["train"].train_test_split(test_size=0.1)
print(f"✅ Dataset: {len(dataset['train'])} train / {len(dataset['test'])} test samples")

# === Load Model and Resize Head ===
print("🔧 Loading base model...")
model = Wav2Vec2ForCTC.from_pretrained(
    "kylegregory/wav2vec2-bisaya",
    ctc_loss_reduction="mean",
    ctc_zero_infinity=True,
    pad_token_id=processor.tokenizer.pad_token_id,
)

vocab_size = len(processor.tokenizer)
model.lm_head = torch.nn.Linear(model.lm_head.in_features, vocab_size, bias=True)
model.config.vocab_size = vocab_size
print(f"🧠 Tokenizer vocab size: {vocab_size}")
print(f"🧠 Model vocab size: {model.config.vocab_size}")

# === Data Collator ===
class DataCollatorCTCWithPadding:
    def __init__(self, processor, padding=True):
        self.processor = processor
        self.padding = padding

    def __call__(self, features):
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, padding=self.padding, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, padding=self.padding, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch["input_ids"] == self.processor.tokenizer.pad_token_id, -100)
        batch["labels"] = labels
        return batch

data_collator = DataCollatorCTCWithPadding(processor)

# === Evaluation Metric ===
def compute_metrics(pred):
    pred_ids = torch.argmax(torch.tensor(pred.predictions), dim=-1)
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    print("\n🔍 Live Eval Debug:")
    for ref, hyp in list(zip(label_str, pred_str))[:3]:
        if "[UNK]" not in ref and "[UNK]" not in hyp:
            print("🧾 REF:", ref)
            print("🔊 HYP:", hyp)

    error_rate = wer(label_str, pred_str)
    os.makedirs("docs", exist_ok=True)
    with open("docs/validation_metrics.md", "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat()} - WER: {error_rate:.4f}\n")

    return {"wer": error_rate}

# === Training Args ===
training_args = TrainingArguments(
    output_dir=f"models/wav2vec2/{MODEL_VERSION}",
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=1000,
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

# === Trainer RNG Patch ===
original_load_rng_state = Trainer._load_rng_state

def patched_load_rng_state(self, checkpoint_path):
    rng_file = os.path.join(checkpoint_path, "rng_state.pth")
    if os.path.exists(rng_file):
        try:
            with torch.serialization.safe_globals(["numpy.core.multiarray._reconstruct"]):
                return original_load_rng_state(self, checkpoint_path)
        except Exception as e:
            print(f"⚠️ Failed to load RNG state from {rng_file}: {e}")
            print("🚨 Training is still safe but not fully reproducible.")
    return None

Trainer._load_rng_state = patched_load_rng_state

# === Trainer Setup ===
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

# === Resume Model Weights If Available ===
print("🚀 Starting Round 2 fine-tuning...")
try:
    checkpoint_root = f"models/wav2vec2/{MODEL_VERSION}"
    checkpoints = sorted([
        d for d in os.listdir(checkpoint_root)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(checkpoint_root, d))
    ], key=lambda x: int(x.split("-")[1]), reverse=True)

    checkpoint_path = None
    for ckpt in checkpoints:
        potential_path = os.path.join(checkpoint_root, ckpt, "pytorch_model.bin")
        if os.path.exists(potential_path):
            checkpoint_path = os.path.join(checkpoint_root, ckpt)
            break

    if checkpoint_path:
        print(f"🔁 Loading weights from: {checkpoint_path}")
        model.load_state_dict(torch.load(os.path.join(checkpoint_path, "pytorch_model.bin")), strict=False)
    else:
        print("🚀 No valid checkpoints found. Starting from base model.")

    # === Looped Training (5 Epochs per Cycle) ===
    EPOCHS_PER_SESSION = 5
    MAX_EPOCHS = 30  # total target

    # Check if a previous training run exists (manual resume from weights only)
    completed_epochs = 0

    while completed_epochs < MAX_EPOCHS:
        remaining_epochs = MAX_EPOCHS - completed_epochs
        epochs_this_round = min(EPOCHS_PER_SESSION, remaining_epochs)

        print(f"🚀 Training for {epochs_this_round} epoch(s)... (Completed so far: {completed_epochs})")
        trainer.args.num_train_epochs = completed_epochs + epochs_this_round  # absolute, not relative

        trainer.train(resume_from_checkpoint=False)

        completed_epochs += epochs_this_round

        # Prompt user whether to continue
        print(f"\n⏹ Completed {completed_epochs}/{MAX_EPOCHS} epochs.")
        response = input("🟢 Continue training another 5 epochs? (y/n): ").strip().lower()
        if response not in ["y", "yes"]:
            print("🛑 Training stopped by user.")
            break

except PermissionError as e:
    print(f"❌ Caught PermissionError: {e}")
    temp_path = str(e).split("'")[1]
    dest_path = str(e).split("'")[3]
    print(f"🔁 Moving manually: {temp_path} → {dest_path}")
    try:
        if os.path.exists(dest_path):
            shutil.rmtree(dest_path)
        shutil.move(temp_path, dest_path)
        print("✅ Manual move complete.")
    except Exception as move_err:
        print(f"❌ Move failed: {move_err}")
        raise

# === Save & Push Final Model ===
trainer.save_model(f"models/wav2vec2/{MODEL_VERSION}")
processor.save_pretrained(f"models/wav2vec2/{MODEL_VERSION}")
print(f"✅ Model saved to models/wav2vec2/{MODEL_VERSION}")

print("📤 Uploading best checkpoint to Hugging Face...")
from scripts.auto_push_checkpoint import push_checkpoint
push_checkpoint(model_dir=f"models/wav2vec2/{MODEL_VERSION}", tag=MODEL_VERSION)
print("🎉 Upload complete.")
