# scripts/loss_plot_callback.py

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

        # Save loss history
        os.makedirs("logs", exist_ok=True)
        df = pd.DataFrame(self.losses)
        df.to_csv("logs/loss_history.csv", index=False)

        # Live plot update
        if len(self.losses) >= 2:
            plt.figure(figsize=(10, 4))
            plt.plot(df["step"], df["loss"], label="Training Loss", color="crimson")
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.title("Live Training Loss")
            plt.grid(True)
            plt.legend()
            os.makedirs("docs", exist_ok=True)
            plt.savefig("docs/loss_plot.png")
            plt.close()
