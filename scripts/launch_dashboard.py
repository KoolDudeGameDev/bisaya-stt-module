import subprocess
import threading
import time
import webbrowser
import os

# === CONFIGURATION ===
SCRIPT_NAME = "scripts/monitor_dashboard.py"  # Path to your Streamlit dashboard
PORT = 8501
NGROK_PATH = "ngrok"  # Or full path like: "C:/ngrok/ngrok.exe"
REGION = "ap"         # e.g., us, eu, ap, au, sa, jp, in

def start_streamlit():
    """Start the Streamlit dashboard in the background."""
    print("🚀 Launching Streamlit dashboard...")
    subprocess.Popen(["streamlit", "run", SCRIPT_NAME], stdout=subprocess.DEVNULL)

def start_ngrok():
    """Start Ngrok and capture public URL."""
    print("🌐 Starting Ngrok tunnel...")
    ngrok_proc = subprocess.Popen(
        [NGROK_PATH, "http", f"{PORT}", "--region", REGION],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    # Wait for Ngrok to initialize
    time.sleep(3)
    print("⏳ Waiting for Ngrok to generate a public URL...")

    # Use Ngrok's API to get the URL
    import requests
    while True:
        try:
            response = requests.get("http://localhost:4040/api/tunnels")
            tunnel_info = response.json()
            public_url = tunnel_info['tunnels'][0]['public_url']
            print(f"🌍 Public URL: {public_url}")
            webbrowser.open(public_url)
            break
        except Exception:
            time.sleep(1)

def main():
    threading.Thread(target=start_streamlit).start()
    time.sleep(5)  # Let Streamlit initialize
    start_ngrok()

if __name__ == "__main__":
    main()
