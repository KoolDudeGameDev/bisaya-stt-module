from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from huggingface_hub import login, HfApi
import os
import sys
import shutil
from datetime import datetime

# Your repo ID
repo_id = "kylegregory/wav2vec2-bisaya"

# 1️⃣ Retrieve token from environment variable
token = os.getenv("HUGGINGFACE_TOKEN")

if not token:
    print("[❌] HUGGINGFACE_TOKEN environment variable not set. Please set it before running.")
    sys.exit(1)

# 2️⃣ Login automatically
login(token=token)

# 3️⃣ Create repo if it doesn't exist (idempotent)
api = HfApi()
try:
    api.create_repo(repo_id, repo_type="model", exist_ok=True, token=token)
    print(f"[ℹ️] Repo '{repo_id}' ensured to exist.")
except Exception as e:
    print(f"[❌] Failed to create or access repo '{repo_id}': {e}")
    sys.exit(1)

# 4️⃣ Load model and processor from local directory
print("[ℹ️] Loading model and processor...")
model = Wav2Vec2ForCTC.from_pretrained("./wav2vec2-bisaya")
processor = Wav2Vec2Processor.from_pretrained("./wav2vec2-bisaya")

# 5️⃣ Save to a clean temp directory
export_dir = "./wav2vec2-bisaya-export"

# Remove if it exists
if os.path.exists(export_dir):
    shutil.rmtree(export_dir)

print(f"[ℹ️] Saving model and processor to '{export_dir}'...")
model.save_pretrained(export_dir, safe_serialization=True)
processor.save_pretrained(export_dir)

# 6️⃣ Generate a version tag based on timestamp
version_tag = datetime.now().strftime("v%Y%m%d-%H%M%S")

# 7️⃣ Upload the folder as a snapshot with commit message and tag
print("[ℹ️] Uploading model and processor to Hugging Face as a versioned snapshot...")
api.upload_folder(
    repo_id=repo_id,
    folder_path=export_dir,
    repo_type="model",
    commit_message=f"Model update {version_tag}",
    token=token,
)

print(f"[✅] Model snapshot '{version_tag}' uploaded successfully.")
print(f"[🎯] View it here: https://huggingface.co/{repo_id}")
