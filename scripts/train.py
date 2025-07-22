from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, TrainingArguments, Trainer
from datasets import load_from_disk
import torch, os, shutil, json
from datetime import datetime
from jiwer import wer
from train_callbacks import LiveSampleLogger
from loss_plot_callback import LossPlotCallback
from huggingface_hub import snapshot_download
import pandas as pd

torch.set_num_threads(os.cpu_count())

# === Config ===
DATASET_VERSION = "v1_training_ready_grapheme"
PROCESSOR_VERSION = "v1_grapheme"
MODEL_VERSION = "v1_bisaya"
HF_REPO_ID = "kylegregory/wav2vec2-bisaya"

# === Load Dataset ===
print("🔍 Loading dataset and processor...")
raw_dataset = load_from_disk(f"data/processed/{DATASET_VERSION}")
processor = Wav2Vec2Processor.from_pretrained(f"processor/{PROCESSOR_VERSION}")

MAX_INPUT_LENGTH_SEC = 15
max_len = int(processor.feature_extractor.sampling_rate * MAX_INPUT_LENGTH_SEC)
filtered_dataset = raw_dataset.filter(lambda x: len(x["input_values"]) <= max_len)

dataset = filtered_dataset["train"].train_test_split(test_size=0.1)
print(f"✅ Dataset: {len(dataset['train'])} train / {len(dataset['test'])} test samples")

# === Detect Local or HF Resume Point ===
print("🔧 Loading model...")
checkpoint_root = f"models/wav2vec2/{MODEL_VERSION}"
checkpoints = sorted([
    d for d in os.listdir(checkpoint_root)
    if d.startswith("checkpoint-") and os.path.isdir(os.path.join(checkpoint_root, d))
], key=lambda x: int(x.split("-")[1]), reverse=True)

checkpoint_path = None
completed_epochs = 0

for ckpt in checkpoints:
    trainer_state_path = os.path.join(checkpoint_root, ckpt, "trainer_state.json")
    if os.path.exists(trainer_state_path):
        checkpoint_path = os.path.join(checkpoint_root, ckpt)
        with open(trainer_state_path, "r", encoding="utf-8") as f:
            state = f.read()
            completed_epochs = int(float(json.loads(state).get("epoch", 0)))
        break

if checkpoint_path:
    print(f"🔁 Resuming from checkpoint: {checkpoint_path}")
    print(f"🔢 Last completed epoch: {completed_epochs}")
    model_dir_to_load = checkpoint_path
else:
    print("📥 Downloading latest pushed model from Hugging Face...")
    model_dir_to_load = snapshot_download(HF_REPO_ID)

model = Wav2Vec2ForCTC.from_pretrained(
    model_dir_to_load,
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
WER_HISTORY = {"step": [], "wer": []}

def compute_metrics(pred):
    global WER_HISTORY

    step = trainer.state.global_step
    pred_ids = torch.argmax(torch.tensor(pred.predictions), dim=-1)
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    error_rate = wer(label_str, pred_str)

    os.makedirs("logs", exist_ok=True)
    WER_HISTORY["step"].append(step)
    WER_HISTORY["wer"].append(error_rate)
    pd.DataFrame(WER_HISTORY).to_csv("logs/val_wer_history.csv", index=False)

    # Live debug
    print("\n🔍 Live Eval Debug:")
    for ref, hyp in list(zip(label_str, pred_str))[:3]:
        if "[UNK]" not in ref and "[UNK]" not in hyp:
            print("📜 REF:", ref)
            print("🔊 HYP:", hyp)

    print(f"📉 Validation WER @ Step {step}: {error_rate:.4f}")

    # Overfitting check
    if len(WER_HISTORY["wer"]) >= 2:
        prev_wer = WER_HISTORY["wer"][-2]
        curr_wer = WER_HISTORY["wer"][-1]
        if curr_wer > prev_wer + 0.02:
            print(f"⚠️ WER increased from {prev_wer:.4f} to {curr_wer:.4f} — possible overfitting.")

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
    num_train_epochs=10,
    learning_rate=5e-5,
    lr_scheduler_type="linear",
    warmup_steps=500,
    logging_steps=50,
    logging_dir="./logs",
    logging_first_step=True,
    save_safetensors=True,
    overwrite_output_dir=True,
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

# === Loop Training ===
print("🚀 Starting Round 2 fine-tuning...")
try:
    EPOCHS_PER_SESSION = 5
    MAX_EPOCHS = 10

    while completed_epochs < MAX_EPOCHS:
        remaining_epochs = MAX_EPOCHS - completed_epochs
        epochs_this_round = min(EPOCHS_PER_SESSION, remaining_epochs)

        print(f"\n🚀 Training for {epochs_this_round} epoch(s)... (Completed so far: {completed_epochs})")
        trainer.args.num_train_epochs = completed_epochs + epochs_this_round

        trainer.train(resume_from_checkpoint=checkpoint_path if completed_epochs == 0 else None)

        completed_epochs += epochs_this_round
        checkpoint_path = None

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
print("💾 Saving final model...")
trainer.save_model(f"models/wav2vec2/{MODEL_VERSION}")
processor.save_pretrained(f"models/wav2vec2/{MODEL_VERSION}")

# === Final Metrics Summary ===
try:
    loss_df = pd.read_csv("logs/loss_history.csv")
    wer_df = pd.read_csv("logs/val_wer_history.csv")

    final_loss = loss_df["loss"].iloc[-1]
    best_wer = wer_df["wer"].min()
    print(f"📊 Final Training Loss: {final_loss:.4f}")
    print(f"🏆 Best Validation WER: {best_wer:.4f}")
except Exception as e:
    print(f"⚠️ Could not summarize final metrics: {e}")

# === Upload to Hugging Face ===
print("📤 Uploading best checkpoint to Hugging Face...")
from scripts.auto_push_checkpoint import push_checkpoint
push_checkpoint(model_dir=f"models/wav2vec2/{MODEL_VERSION}", tag=MODEL_VERSION)
print("🎉 Upload complete.")
