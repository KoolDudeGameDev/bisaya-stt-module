import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from stt_ui import record_and_transcribe

# === CONFIG ===
METRICS_FILE = "docs/validation_metrics.md"
LOSS_LOG = "logs/loss_history.csv"
WER_LOG = "logs/val_wer_history.csv"
CHECKPOINT_DIR = "models/wav2vec2/v1_bisaya"
TRAIN_SIZE = 3080
BATCH_SIZE = 1
GRAD_ACCUM = 4
STEPS_PER_EPOCH = (TRAIN_SIZE + (BATCH_SIZE * GRAD_ACCUM) - 1) // (BATCH_SIZE * GRAD_ACCUM)

st.set_page_config(page_title="STT Dashboard", layout="wide")
st.title("📡 Bisaya STT Training Monitor")

# === Metric Plots ===
col1, col2 = st.columns(2)

def plot_csv(log_path, y_label, smoothing=5):
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        df["smooth"] = df.iloc[:, 1].rolling(smoothing).mean()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df.iloc[:, 0], df.iloc[:, 1], alpha=0.3, label="Raw")
        ax.plot(df.iloc[:, 0], df["smooth"], label="Smoothed", linewidth=2)
        ax.set_title(y_label)
        ax.set_xlabel("Step")
        ax.set_ylabel(y_label)
        ax.grid(True)
        ax.legend()
        return fig
    return None

with col1:
    st.subheader("📉 Training Loss")
    fig_loss = plot_csv(LOSS_LOG, "Loss")
    if fig_loss: st.pyplot(fig_loss)
    else: st.warning("No loss log found.")

with col2:
    st.subheader("📊 Validation WER")
    fig_wer = plot_csv(WER_LOG, "WER")
    if fig_wer: st.pyplot(fig_wer)
    else: st.warning("No WER log found.")

# === Checkpoint Progress ===
st.markdown("## 🔍 Checkpoint Status")
def get_latest_checkpoint():
    if not os.path.exists(CHECKPOINT_DIR):
        return None
    import re
    checkpoints = [int(m.group(1)) for m in (re.match(r"checkpoint-(\d+)", d) for d in os.listdir(CHECKPOINT_DIR)) if m]
    return max(checkpoints) if checkpoints else None

latest_step = get_latest_checkpoint()
if latest_step:
    estimated_epoch = latest_step / STEPS_PER_EPOCH
    st.metric("🧱 Latest Step", value=latest_step)
    st.metric("📈 Approx. Epoch", value=f"{estimated_epoch:.2f}")
else:
    st.error("No checkpoints detected.")

# === STT Inference UI ===
st.divider()
st.header("🧠 Try the Model")
record_and_transcribe()

# === Auto Refresh ===
st_autorefresh = st.empty()
refresh_rate = st.sidebar.slider("🔁 Refresh interval (seconds)", 10, 300, 60)
st.sidebar.info("Live dashboard auto-refreshes.")

st_autorefresh.markdown(f"""
    <script>
        setTimeout(() => window.location.reload(), {refresh_rate * 1000});
    </script>
""", unsafe_allow_html=True)
