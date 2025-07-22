from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from huggingface_hub import login, HfApi
from datetime import datetime
import argparse, os, shutil, json

def update_model_card(export_dir, tag, dataset_version, vocab_size, num_samples, wer_score):
    template_path = "scripts/model_card_template.md"
    output_path = os.path.join(export_dir, "README.md")
    with open(template_path, "r", encoding="utf-8") as f:
        content = f.read()
    content = content.replace("{{version_tag}}", tag)
    content = content.replace("{{timestamp}}", datetime.now().strftime("%Y-%m-%d %H:%M"))
    content = content.replace("{{dataset_version}}", dataset_version)
    content = content.replace("{{vocab_size}}", str(vocab_size))
    content = content.replace("{{num_samples}}", str(num_samples))
    content = content.replace("{{wer}}", f"{wer_score:.4f}" if wer_score else "N/A")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    print("📄 README.md updated.")

def push_checkpoint(model_dir: str, tag: str, repo_id: str):
    token = os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        print("[❌] HUGGINGFACE_TOKEN not set in environment. Aborting.")
        return
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

    model = Wav2Vec2ForCTC.from_pretrained(model_dir)
    processor = Wav2Vec2Processor.from_pretrained(model_dir)
    model.save_pretrained(export_dir, safe_serialization=True)
    processor.save_pretrained(export_dir)

    try:
        with open("docs/last_wer.json", "r") as f:
            wer_score = json.load(f)["wer"]
    except:
        wer_score = None

    update_model_card(
        export_dir,
        tag=tag,
        dataset_version="v1_training_ready_grapheme",
        vocab_size=len(processor.tokenizer),
        num_samples=0,
        wer_score=wer_score,
    )

    version_tag = datetime.now().strftime(f"{tag}-%Y%m%d-%H%M")
    try:
        api.upload_folder(
            repo_id=repo_id,
            folder_path=export_dir,
            repo_type="model",
            commit_message=f"Auto checkpoint push: {version_tag}",
            token=token,
        )
        print(f"[✅] Checkpoint pushed successfully: {version_tag}")
    except Exception as e:
        print(f"[❌] Upload failed: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--repo", default="kylegregory/wav2vec2-bisaya")
    args = parser.parse_args()
    push_checkpoint(model_dir=args.model_dir, tag=args.tag, repo_id=args.repo)