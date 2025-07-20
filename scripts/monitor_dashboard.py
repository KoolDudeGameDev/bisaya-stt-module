import os
import time
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# === CONFIGURATION ===
METRICS_FILE = "docs/validation_metrics.md"
LOSS_LOG = "logs/loss_history.csv"
CHECKPOINT_DIR = "models/wav2vec2/v1_bisaya"
TRAIN_SIZE = 3080
BATCH_SIZE = 1
GRAD_ACCUM = 4
STEPS_PER_EPOCH = (TRAIN_SIZE + (BATCH_SIZE * GRAD_ACCUM) - 1) // (BATCH_SIZE * GRAD_ACCUM)

# === PAGE SETUP ===
st.set_page_config(page_title="STT Training Monitor", layout="wide")
st.title("Bisaya STT Model Training Monitor")

# === FUNCTION: Latest WER Metrics ===
def show_latest_wer(md_path=METRICS_FILE):
    if not os.path.exists(md_path):
        st.warning("`validation_metrics.md` not found yet.")
        return
    with open(md_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        if not lines:
            st.info("No WER data yet.")
            return
        st.subheader("🧾 Latest WER Metrics")
        st.code("".join(lines[-5:]))

# === FUNCTION: Plot Loss ===
def plot_loss(log_path=LOSS_LOG):
    if not os.path.exists(log_path):
        st.warning("`loss_history.csv` not found yet.")
        return
    df = pd.read_csv(log_path)
    if df.empty:
        st.info("No loss data available.")
        return
    df["smoothed_loss"] = df["loss"].rolling(5).mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["step"], df["loss"], label="Raw Loss", alpha=0.4)
    ax.plot(df["step"], df["smoothed_loss"], label="Smoothed Loss", linewidth=2)
    ax.set_title("Training Loss Over Steps")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

# === FUNCTION: Checkpoint Progress ===
def get_latest_checkpoint(path=CHECKPOINT_DIR):
    if not os.path.exists(path):
        return None
    import re
    checkpoint_steps = []
    for d in os.listdir(path):
        match = re.match(r"checkpoint-(\d+)", d)
        if match:
            checkpoint_steps.append(int(match.group(1)))
    return max(checkpoint_steps) if checkpoint_steps else None

def show_checkpoint_status():
    current_step = get_latest_checkpoint()
    if current_step is None:
        st.error("No checkpoints found.")
        return
    estimated_epoch = current_step / STEPS_PER_EPOCH
    st.metric(label="🧱 Latest Checkpoint", value=f"Step {current_step}")
    st.metric(label="📊 Estimated Epoch", value=f"{estimated_epoch:.2f}")

# === LAYOUT ===
col1, col2 = st.columns(2)

with col1:
    show_latest_wer()

with col2:
    show_checkpoint_status()

st.divider()
plot_loss()

# === AUTO REFRESH ===
st_autorefresh = st.empty()
refresh_rate = st.sidebar.slider("Refresh interval (seconds)", 10, 300, 60)
st.sidebar.info("Page will auto-refresh based on this interval.")

# Auto-refresh with JS injection
st_autorefresh.markdown(f'''
    <script>
    function refresh() {{
        window.location.reload();
    }}
    setTimeout(refresh, {refresh_rate * 1000});
    </script>
''', unsafe_allow_html=True)
