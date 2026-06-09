import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from RQ_experiments.utils import generate_rq

if __name__ == "__main__":
    DATA_FILE = "lm_eval/results/partial_results_codegreen_.jsonl"
    REPORT_DIR = Path(__file__).parent / "report"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    
    generate_rq(DATA_FILE, str(REPORT_DIR), "rq1")
