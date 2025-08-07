import yaml
import subprocess
from lm_eval.configs import YAML_RUN_CONFIG
from lm_eval.utils import simple_parse_args_string
import sys
import os


def main_evaluate():
    with open(YAML_RUN_CONFIG, "r") as f:
        cfg = yaml.safe_load(f)
    

    for model_arg in cfg["model_args"]:
        model_name = simple_parse_args_string(model_arg)["pretrained"]
        cmd = [sys.executable, "download_models.py", "--model_name", model_name]
        result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))

        
    for task in cfg["tasks"]:
        for model_arg, bnb_config in zip(cfg["model_args"], cfg["bnb_config"]):
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
                cfg["experiments_run"]
            ]
            if bnb_config:
                cmd += ["--bnb_config", bnb_config]
            
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"Command failed with exit code {result.returncode}")
                break
            print()


if __name__ == "__main__":
    main_evaluate()
