import subprocess
import threading
import time
import webbrowser
import os

# === CONFIGURATION ===
SCRIPT_NAME = "scripts/dashboard/monitor_dashboard.py"
PORT = 8501
NGROK_PATH = "ngrok"  # Change this if using a full path
REGION = "ap"

def start_streamlit():
    print("🚀 Launching Streamlit dashboard...")
    subprocess.Popen(["streamlit", "run", SCRIPT_NAME], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def start_ngrok():
    print("🌐 Starting Ngrok tunnel...")
    ngrok_proc = subprocess.Popen(
        [NGROK_PATH, "http", str(PORT), "--region", REGION],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    time.sleep(5)
    print("⏳ Fetching Ngrok public URL...")

    import requests
    while True:
        try:
            response = requests.get("http://localhost:4040/api/tunnels")
            tunnels = response.json()["tunnels"]
            public_url = tunnels[0]["public_url"]
            print(f"🌍 Public URL: {public_url}")
            webbrowser.open(public_url)
            break
        except Exception:
            time.sleep(1)

def main():
    threading.Thread(target=start_streamlit).start()
    time.sleep(5)
    start_ngrok()

if __name__ == "__main__":
    main()
