from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from huggingface_hub import login, HfApi
from datetime import datetime
import os
import shutil
import sys

def push_checkpoint(model_dir: str, tag: str):
    repo_id = "kylegregory/wav2vec2-bisaya"
    token = os.getenv("HUGGINGFACE_TOKEN")

    if not token:
        print("[❌] HUGGINGFACE_TOKEN not set.")
        return

    login(token=token)
    api = HfApi()
    try:
        api.create_repo(repo_id, repo_type="model", exist_ok=True, token=token)
    except Exception as e:
        print(f"[❌] Could not access repo: {e}")
        return

    export_dir = f"{model_dir}-export"
    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)

    model = Wav2Vec2ForCTC.from_pretrained(model_dir)
    processor = Wav2Vec2Processor.from_pretrained(model_dir)
    model.save_pretrained(export_dir, safe_serialization=True)
    processor.save_pretrained(export_dir)

    version_tag = datetime.now().strftime(f"{tag}-%Y%m%d-%H%M")
    api.upload_folder(
        repo_id=repo_id,
        folder_path=export_dir,
        repo_type="model",
        commit_message=f"Auto checkpoint push: {version_tag}",
        token=token,
    )

    print(f"[✅] Checkpoint pushed: {version_tag}")
