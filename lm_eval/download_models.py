from transformers import AutoModel, AutoTokenizer, AutoConfig
from huggingface_hub import snapshot_download
import argparse

def download_and_load_models(model_name: str) -> None:
    print(f"Fetching {model_name}…")
    # this will ensure all files in model cards are stored into our local folder
    snapshot_download(repo_id=model_name)
    # sometimes downloading alone doesn't fetch everything,
    # we need to load the model to completely get safetensor and other things
    _ = AutoConfig.from_pretrained(model_name)
    _ = AutoModel.from_pretrained(model_name)
    _ = AutoTokenizer.from_pretrained(model_name)
    print(f"Done: {model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="downloading models")
    parser.add_argument('--model_name', type=str, help='pretrained model name from huggingface')
    args = parser.parse_args()
    models = download_and_load_models(args.model_name)

