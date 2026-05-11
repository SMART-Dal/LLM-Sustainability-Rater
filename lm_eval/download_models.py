from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, AutoConfig
from huggingface_hub import snapshot_download
import argparse


def download_and_load_models(model_name: str) -> None:
    print(f"Fetching {model_name}…")
    # this will ensure all files in model cards are stored into our local folder
    snapshot_download(repo_id=model_name)
    # sometimes downloading alone doesn't fetch everything,
    # we need to load the model to completely get safetensor and other things
    _ = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    _ = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    try:
        # First, try loading it as a text-generation LLM (fixes Replit/CodeShell)
        _ = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    except ValueError:
        # If it throws a ValueError, it's likely a base model or embedding model. 
        # Fall back to standard AutoModel.
        print(f"[{model_name}] is not a CausalLM. Falling back to base AutoModel...")
        _ = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    print(f"Done: {model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="downloading models")
    parser.add_argument(
        "--model_name", type=str, help="pretrained model name from huggingface"
    )
    args = parser.parse_args()
    models = download_and_load_models(args.model_name)
