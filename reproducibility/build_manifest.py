"""Build the reproducibility manifests for the paper's CodeGreen results.

Writes:
  reproducibility/manifest.json           exact HF revision, license and #params of every model and dataset
  lm_eval/rating_llms/model_sizes.json    frozen model sizes (GiB) used by the RQ4 size-efficiency ratings

Revisions: `main_run.py` calls `snapshot_download(repo)` (the repo's `main` branch) right before each run,
so the revision used is the last commit on `main` at or before the run. Run times come from the `timestamp`
of each row in lm_eval/results/final_results_codegreen.jsonl.

Sizes: same rule as `lm_eval.rating_llms.utils.utils.get_model_size_gb` (total size of the *.safetensors
files, else *.bin files, at that revision; if the repo is unavailable, a name-based FP16 estimate).

Usage (from the repo root):  python reproducibility/build_manifest.py
"""
import json
import re
import struct
from datetime import datetime
from pathlib import Path

import torch
from huggingface_hub import HfApi, constants
from huggingface_hub.errors import RepositoryNotFoundError
from transformers import AutoConfig, AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "lm_eval" / "results" / "final_results_codegreen.jsonl"
MANIFEST = ROOT / "reproducibility" / "manifest.json"
SIZES = ROOT / "lm_eval" / "rating_llms" / "model_sizes.json"
DATASETS = {  # task name in results -> (HF dataset repo, how the task yaml uses it)
    "livecodebench": ("livecodebench/code_generation_lite", {"version_tag": "release_v6", "split": "test"}),
    "code2text_python": ("CM/codexglue_code2text_python", {"split": "test", "examples": "first 500"}),
}
# license fields that the HF card leaves as a bare "other" or empty (taken from the model/dataset card text)
LICENSE_NOTES = {
    "stabilityai/stable-code-3b": "other (Stability AI Community License)",
    "livecodebench/code_generation_lite": "cc (variant not specified on the card)",
    "CM/codexglue_code2text_python": "C-UDA (CodeXGLUE data license; source: CodeSearchNet)",
}
api = HfApi()


def revision_at(repo, when, repo_type="model"):
    """Latest commit on `main` created at or before `when`."""
    for c in api.list_repo_commits(repo, repo_type=repo_type):  # newest first
        if c.created_at <= when:
            return c.commit_id, c.created_at.isoformat()
    raise RuntimeError(f"no commit of {repo} before {when}")


def license_of(repo, info):
    if repo in LICENSE_NOTES:
        return LICENSE_NOTES[repo]
    card = info.card_data.to_dict() if info.card_data else {}
    lic = card.get("license")
    return f"other ({card['license_name']})" if lic == "other" and card.get("license_name") else lic


def size_gib(files):
    """Same rule as get_model_size_gb: *.safetensors total, else *.bin total."""
    st = [v for k, v in files.items() if k.endswith(".safetensors") and v]
    return (sum(st) if st else sum(v for k, v in files.items() if k.endswith(".bin") and v)) / 1024**3


def name_estimate_gib(name):
    """Same fallback as get_model_size_gb when the repo can't be queried: '<N>b'/'<N>m' in the name, FP16."""
    value, unit = re.search(r"(?i)(\d+(?:\.\d+)?)([bm])\b", name).groups()
    return float(value) * (1e9 if unit.lower() == "b" else 1e6) * 2 / 1024**3


def params_from_config(name, rev):
    """Parameter count for checkpoints without safetensors metadata (.bin only), counted on the meta device."""
    cfg = AutoConfig.from_pretrained(name, revision=rev)
    with torch.device("meta"):
        return AutoModelForCausalLM.from_config(cfg).num_parameters()


def from_local_cache(name):
    """Fallback for models removed from the Hub: revision/license/params from the local HF cache."""
    repo_dir = Path(constants.HF_HUB_CACHE) / f"models--{name.replace('/', '--')}"
    rev = (repo_dir / "refs" / "main").read_text().strip()
    snap = repo_dir / "snapshots" / rev
    lic = next((l.split(":", 1)[1].strip() for l in (snap / "README.md").read_text().splitlines()
                if l.startswith("license:")), None)
    params = 0
    for f in set(json.loads((snap / "model.safetensors.index.json").read_text())["weight_map"].values()):
        with open(snap / f, "rb") as fh:
            header = json.loads(fh.read(struct.unpack("<Q", fh.read(8))[0]))
        for k, v in header.items():
            if k != "__metadata__":
                n = 1
                for d in v["shape"]:
                    n *= d
                params += n
    return rev, lic, params


rows = [json.loads(l) for l in open(RESULTS)]
runs = {}  # model -> {task: run timestamp}
for r in rows:
    runs.setdefault(r["model"], {})[r["task_name"]] = datetime.fromisoformat(r["timestamp"])

models, sizes = [], {}
for name, task_times in runs.items():
    try:
        revs = {t: revision_at(name, ts) for t, ts in task_times.items()}
        if len({rev for rev, _ in revs.values()}) != 1:
            raise RuntimeError(f"{name}: revision changed between tasks: {revs}")
        rev, rev_date = next(iter(revs.values()))
        info = api.model_info(name, revision=rev, files_metadata=True)
        params = info.safetensors.total if info.safetensors else params_from_config(name, rev)
        entry = {"revision": rev, "revision_date": rev_date, "license": license_of(name, info),
                 "params": params, "source": "hub"}
        size, size_note = size_gib({s.rfilename: s.size for s in info.siblings}), "weight files at revision"
    except RepositoryNotFoundError:
        rev, lic, params = from_local_cache(name)
        entry = {"revision": rev, "revision_date": None, "license": lic, "params": params,
                 "source": "local HF cache (repository no longer available on the Hub)"}
        size, size_note = name_estimate_gib(name), "FP16 estimate from model name (repo unavailable on the Hub)"
    models.append({"model": name, **entry, "tasks": sorted(task_times)})
    sizes[name] = {"size_gb": size, "params": params, "revision": rev, "size_source": size_note}
    print(f"{name:55s} {rev[:10]} {str(entry['revision_date'])[:10]} {entry['license']} "
          f"params={params} size={size:.3f}GiB")

datasets = []
for task, (repo, cfg) in DATASETS.items():
    first_run = min(ts for m in runs.values() for t, ts in m.items() if t == task)
    rev, rev_date = revision_at(repo, first_run, repo_type="dataset")
    info = api.dataset_info(repo, revision=rev)
    datasets.append({"task": task, "dataset": repo, "revision": rev, "revision_date": rev_date,
                     "config": cfg, "license": license_of(repo, info)})
    print(f"{repo:55s} {rev[:10]} {rev_date[:10]} {datasets[-1]['license']}")

MANIFEST.write_text(json.dumps({
    "manifest_version": "1.0",
    "results_file": str(RESULTS.relative_to(ROOT)),
    "experiments_run": sorted({r["experiments_run"] for r in rows}),
    "run_period": [min(r["timestamp"] for r in rows), max(r["timestamp"] for r in rows)],
    "environment": "reproducibility/requirements.lock.txt (Python 3.10, Linux x86_64)",
    "model_sizes": str(SIZES.relative_to(ROOT)),
    "models": models,
    "datasets": datasets,
}, indent=2) + "\n")
SIZES.write_text(json.dumps({
    "manifest_version": "1.0",
    "unit": "GiB (total size of the model's weight files: *.safetensors, else *.bin)",
    "used_by": "lm_eval/rating_llms/utils/utils.py:get_model_size_gb (RQ4 size_acc / size_ene)",
    "models": sizes,
}, indent=2) + "\n")
print(f"wrote {MANIFEST.relative_to(ROOT)} ({len(models)} models, {len(datasets)} datasets) "
      f"and {SIZES.relative_to(ROOT)}")
