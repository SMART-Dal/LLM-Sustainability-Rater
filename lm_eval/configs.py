from pathlib import Path


MAIN_DIR = Path(__file__).parent
OUTPUT_DIR = MAIN_DIR / "results"
OUTPUT_FILE = OUTPUT_DIR / "results.jsonl"
CODE_CARBON_LOG_DIR = MAIN_DIR / "codecarbon_log"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CODE_CARBON_LOG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE.touch(exist_ok=True)


YAML_RUN_CONFIG = MAIN_DIR / "run_config.yaml"

DOTENV_FILE = MAIN_DIR.parent / ".env"
