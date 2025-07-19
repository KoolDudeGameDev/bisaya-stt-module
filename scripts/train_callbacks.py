from transformers import TrainerCallback
import torch

class LiveSampleLogger(TrainerCallback):
    def __init__(self, processor, eval_dataset, sample_count=3):
        self.processor = processor
        self.eval_dataset = eval_dataset.select(range(sample_count))

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        model = kwargs["model"].eval().cpu()

        for i, sample in enumerate(self.eval_dataset):
            input_values = torch.tensor(sample["input_values"]).unsqueeze(0)
            with torch.no_grad():
                logits = model(input_values).logits
                pred_ids = torch.argmax(logits, dim=-1)
                transcription = self.processor.batch_decode(pred_ids)[0]

            # Decode labels into reference text
            reference = self.processor.decode(sample["labels"], group_tokens=False)

            print(f"[🔎 Sample {i+1}]")
            print("📌 Reference :", reference)
            print("🧠 Predicted :", transcription)
            print("---")
