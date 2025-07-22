from transformers import TrainerCallback
import matplotlib.pyplot as plt
import pandas as pd
import os

class LossPlotCallback(TrainerCallback):
    def __init__(self):
        self.losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or "loss" not in logs:
            return

        step = state.global_step
        loss = logs["loss"]
        self.losses.append({"step": step, "loss": loss})

        os.makedirs("logs", exist_ok=True)
        df = pd.DataFrame(self.losses)
        df.to_csv("logs/loss_history.csv", index=False)

        # Load WER history if exists
        wer_path = "logs/val_wer_history.csv"
        wer_df = pd.read_csv(wer_path) if os.path.exists(wer_path) else pd.DataFrame()

        if len(self.losses) >= 2:
            plt.figure(figsize=(10, 5))
            plt.plot(df["step"], df["loss"], label="Training Loss", color="crimson")

            if not wer_df.empty:
                plt.plot(wer_df["step"], wer_df["wer"], label="Validation WER", color="blue")

            plt.xlabel("Step")
            plt.ylabel("Loss / WER")
            plt.title("Training Loss & Validation WER")
            plt.grid(True)
            plt.legend()
            os.makedirs("docs", exist_ok=True)
            plt.savefig("docs/loss_plot.png")
            plt.close()
