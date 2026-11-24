import argparse
import yaml
import subprocess
from lm_eval.configs import YAML_RUN_CONFIG, DOTENV_FILE
from lm_eval.utils import simple_parse_args_string
from lm_eval.download_models import paper_revision
import sys
import os
from dotenv import load_dotenv
from huggingface_hub import login

def load_hf_token():
    load_dotenv(DOTENV_FILE)
    hf_token = os.getenv("HF_TOKEN")

    if hf_token:
        login(token=hf_token)
    else:
        raise ValueError("Hugging Face token not found in environment variables")

def main_evaluate(repro=False):
    with open(YAML_RUN_CONFIG, "r") as f:
        cfg = yaml.safe_load(f)
    

    for model_arg in cfg["model_args"]:
        model_name = simple_parse_args_string(model_arg)["pretrained"]
        cmd = [sys.executable, "download_models.py", "--model_name", model_name]
        if repro:
            cmd.append("--repro")
        result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))

        
    for task in cfg["tasks"]:
        for model_arg, bnb_config in zip(cfg["model_args"], cfg["bnb_config"]):
            # load the downloaded paper revision (without revision=, lm_eval fetches the Hub's latest main)
            if repro and (rev := paper_revision(simple_parse_args_string(model_arg)["pretrained"])):
                model_arg += f",revision={rev}"
            print(f"{'-' * 20} running model: {model_arg}, on task {task} {'-' * 20}")
            cmd = [
                "lm_eval",
                "--model",
                cfg["model"],
                "--model_args",
                model_arg,
                "--tasks",
                task,
                "--device",
                "cuda:0",
                "--confirm_run_unsafe_code",
                "--experiments_run",
                cfg["experiments_run"],
                "--trust_remote_code"
            ]
            if bnb_config:
                cmd += ["--bnb_config", bnb_config]
            
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"Command failed with exit code {result.returncode}")
                continue



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run models on benchmarks with energy profiling")
    parser.add_argument(
        "--repro", action="store_true",
        help="use the exact model revisions of the paper (reproducibility/manifest.json) instead of main",
    )
    cli_args = parser.parse_args()
    load_hf_token()
    main_evaluate(repro=cli_args.repro)
