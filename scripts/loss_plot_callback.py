# scripts/loss_plot_callback.py
from transformers import TrainerCallback
import matplotlib.pyplot as plt
import os

class LossPlotCallback(TrainerCallback):
    def __init__(self):
        self.losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or "loss" not in logs:
            return
        self.losses.append(logs["loss"])
        if len(self.losses) < 2:
            return

        plt.figure(figsize=(10, 5))
        plt.plot(self.losses, label="Training Loss", color="red")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title("Live Training Loss")
        plt.grid(True)
        plt.legend()

        os.makedirs("docs", exist_ok=True)
        plt.savefig("docs/loss_plot.png")
        plt.close()
