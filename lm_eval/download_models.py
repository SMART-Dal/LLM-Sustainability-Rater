from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, AutoConfig
from huggingface_hub import snapshot_download
from pathlib import Path
import argparse
import json

# exact model revisions used in the paper (see reproducibility/build_manifest.py)
MANIFEST = Path(__file__).resolve().parents[1] / "reproducibility" / "manifest.json"


def paper_revision(model_name: str):
    """Commit of `model_name` used in the paper, or None if the model is not in the manifest."""
    models = json.loads(MANIFEST.read_text())["models"]
    return next((m["revision"] for m in models if m["model"] == model_name), None)


def download_and_load_models(model_name: str, revision: str = "main") -> None:
    print(f"Fetching {model_name} (revision: {revision})…")
    # this will ensure all files in model cards are stored into our local folder
    snapshot_download(repo_id=model_name, revision=revision)
    # sometimes downloading alone doesn't fetch everything,
    # we need to load the model to completely get safetensor and other things
    _ = AutoConfig.from_pretrained(model_name, revision=revision, trust_remote_code=True)
    _ = AutoTokenizer.from_pretrained(model_name, revision=revision, trust_remote_code=True)
    try:
        # First, try loading it as a text-generation LLM (fixes Replit/CodeShell)
        _ = AutoModelForCausalLM.from_pretrained(model_name, revision=revision, trust_remote_code=True)
    except ValueError:
        # If it throws a ValueError, it's likely a base model or embedding model.
        # Fall back to standard AutoModel.
        print(f"[{model_name}] is not a CausalLM. Falling back to base AutoModel...")
        _ = AutoModel.from_pretrained(model_name, revision=revision, trust_remote_code=True)
    print(f"Done: {model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="downloading models")
    parser.add_argument(
        "--model_name", type=str, help="pretrained model name from huggingface"
    )
    parser.add_argument(
        "--repro", action="store_true",
        help="download the exact revision used in the paper (reproducibility/manifest.json) instead of main",
    )
    args = parser.parse_args()
    revision = "main"
    if args.repro:
        revision = paper_revision(args.model_name)
        if revision is None:
            print(f"[{args.model_name}] is not in {MANIFEST.name}, downloading main instead.")
            revision = "main"
    models = download_and_load_models(args.model_name, revision)
