# === Base image with Python 3.11 support ===
FROM python:3.11-slim

# === Set working directory ===
WORKDIR /app

# === Copy only dashboard-related files and shared modules ===
COPY scripts/dashboard/ ./dashboard/
COPY scripts/core/loss_plot_callback.py ./core/loss_plot_callback.py
COPY scripts/core/train_callbacks.py ./core/train_callbacks.py
COPY scripts/core/model_card_template.md ./core/model_card_template.md
COPY logs/ ./logs/
COPY docs/ ./docs/

# === Install dependencies ===
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir streamlit pyngrok pandas matplotlib watchdog

RUN pip install pyngrok
ENV NGROK_AUTHTOKEN=30DiPlqTFiG91M8N4NoFrlJ2QC8_4HmTpJhoSHCK1cpUk7ten

# Optional: add `transformers` if UI depends on it
RUN pip install --no-cache-dir transformers==4.41.2

# === Expose Streamlit default port ===
EXPOSE 8501

# === Run Streamlit Dashboard ===
CMD ["streamlit", "run", "dashboard/launch_dashboard.py", "--server.port=8501", "--server.headless=true"]


# === HOW TO USE ===
# docker build -t bisaya-dashboard -f scripts/dashboard/Dockerfile .

# docker run -it --rm -p 8501:8501 --env NGROK_AUTH_TOKEN=30DiPlqTFiG91M8N4NoFrlJ2QC8_4HmTpJhoSHCK1cpUk7ten bisaya-dashboard

# IF it doesnt start automatically
# streamlit run dashboard/launch_dashboard.py
