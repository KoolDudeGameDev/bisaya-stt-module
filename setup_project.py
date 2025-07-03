import os
import sys
import subprocess
import platform

def run(command, cwd=None):
    """Run a shell command and exit if it fails."""
    print(f"🟢 Running: {command}")
    result = subprocess.run(command, shell=True, cwd=cwd)
    if result.returncode != 0:
        print("❌ Command failed. Exiting.")
        sys.exit(1)

def create_venv():
    if platform.system() == "Windows":
        venv_path = "bisaya_stt_venv\\Scripts\\activate"
        if not os.path.exists("bisaya_stt_venv"):
            run("python -m venv bisaya_stt_venv")
        activate_cmd = f"{venv_path}"
    else:
        venv_path = "bisaya_stt_venv/bin/activate"
        if not os.path.exists("bisaya_stt_venv"):
            run("python3 -m venv bisaya_stt_venv")
        activate_cmd = f"source {venv_path}"

    print("✅ Virtual environment created.")
    print(f"⚠️ Please activate it manually with:\n\n   {activate_cmd}\n")
    print("After activation, run:")
    print("   pip install --upgrade pip")
    print("   pip install -r requirements.txt")
    sys.exit(0)

def main():
    if not os.path.exists("bisaya_stt_venv"):
        create_venv()

    print("✅ Virtual environment already exists. Make sure it is activated.")

    # Install dependencies
    run("pip install --upgrade pip")
    run("pip install -r requirements.txt")

    # Prepare datasets
    run("python prepare_dataset.py")
    run("python prepare_training_dataset.py")

    print("\n🎉 Setup complete! You can now train with:\n")
    print("   python train.py")

if __name__ == "__main__":
    main()
