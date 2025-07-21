from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from huggingface_hub import login, HfApi
from datetime import datetime
import argparse
import os
import shutil
import sys

def push_checkpoint(model_dir: str, tag: str, repo_id: str):
    token = os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        print("[❌] HUGGINGFACE_TOKEN not set in environment. Aborting.")
        return

    print(f"🔑 Logging in to Hugging Face Hub as '{repo_id}'...")
    login(token=token)
    api = HfApi()

    try:
        api.create_repo(repo_id, repo_type="model", exist_ok=True, token=token)
    except Exception as e:
        print(f"[❌] Could not access/create repo '{repo_id}': {e}")
        return

    export_dir = f"{model_dir}-export"
    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)

    print("📦 Exporting model and processor...")
    model = Wav2Vec2ForCTC.from_pretrained(model_dir)
    processor = Wav2Vec2Processor.from_pretrained(model_dir)

    model.save_pretrained(export_dir, safe_serialization=True)
    processor.save_pretrained(export_dir)

    version_tag = datetime.now().strftime(f"{tag}-%Y%m%d-%H%M")
    print(f"📤 Uploading to Hugging Face Hub with tag: {version_tag}")

    try:
        api.upload_folder(
            repo_id=repo_id,
            folder_path=export_dir,
            repo_type="model",
            commit_message=f"Auto checkpoint push: {version_tag}",
            token=token,
        )
    except Exception as e:
        print(f"[❌] Upload failed: {e}")
        return

    print(f"[✅] Checkpoint pushed successfully: {version_tag}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push model checkpoint to Hugging Face Hub.")
    parser.add_argument("--model_dir", required=True, help="Directory of the trained model to export.")
    parser.add_argument("--tag", required=True, help="Version tag or label (e.g., v1_bisaya).")
    parser.add_argument("--repo", required=False, default="kylegregory/wav2vec2-bisaya", help="HF repo ID")

    args = parser.parse_args()
    push_checkpoint(model_dir=args.model_dir, tag=args.tag, repo_id=args.repo)
