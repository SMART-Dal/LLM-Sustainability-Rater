"""Build reproducibility/requirements.lock.txt: the full, exact dependency closure of the pins in
pyproject.toml, with one sha256 hash per package, for Python 3.10 on Linux x86_64.

How: pip's resolver is run in dry-run mode on the pinned dependencies (nothing is installed); its JSON
report lists every package it would install with the exact version and file hash. Packages that are
needed with an "extra" (e.g. typepy[datetime]) are listed both with and without the extra, so that
--require-hashes works on old pip versions (tested: pip 22.0.2 and 26.3).

Usage (from the repo root, with Python 3.10 and pip >= 22.2):  python reproducibility/build_lock.py
"""
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "reproducibility" / "requirements.lock.txt"
ENV = {"python_version": "3.10", "python_full_version": "3.10.12", "sys_platform": "linux",
       "platform_system": "Linux", "platform_machine": "x86_64", "implementation_name": "cpython",
       "platform_python_implementation": "CPython", "os_name": "posix"}

pins = re.findall(r'"([^"]+)"', re.search(r"dependencies = \[(.*?)\]", (ROOT / "pyproject.toml").read_text(),
                                            re.S).group(1))
with tempfile.TemporaryDirectory() as tmp:
    reqs, report = Path(tmp) / "pins.txt", Path(tmp) / "report.json"
    reqs.write_text("\n".join(pins) + "\n")
    subprocess.run([sys.executable, "-m", "pip", "install", "--dry-run", "--ignore-installed", "--quiet",
                    "--report", str(report), "-r", str(reqs)], check=True)
    items = json.loads(report.read_text())["install"]

# every extra any package (or a pin) asks for, e.g. typepy -> {"datetime"}
installed = {canonicalize_name(i["metadata"]["name"]) for i in items}
extras = {}
for req in [Requirement(p) for p in pins]:
    extras.setdefault(canonicalize_name(req.name), set()).update(req.extras)
for i in items:
    name = canonicalize_name(i["metadata"]["name"])
    extras.setdefault(name, set()).update(i.get("requested_extras") or [])
    active = [""] + list(i.get("requested_extras") or [])
    for r in i["metadata"].get("requires_dist") or []:
        req = Requirement(r)
        if req.extras and canonicalize_name(req.name) in installed and (
                req.marker is None or any(req.marker.evaluate({**ENV, "extra": e}) for e in active)):
            extras.setdefault(canonicalize_name(req.name), set()).update(req.extras)

lines = ["# Lockfile: full, exact dependency closure of the pins in pyproject.toml.",
         "# Resolved for Python 3.10 on Linux x86_64 (PyPI). Every package is pinned with its sha256.",
         "#",
         "# Install (from the repo root, in a fresh Python 3.10 venv):",
         "#   pip install --require-hashes -r reproducibility/requirements.lock.txt",
         "#   pip install --no-deps -e .",
         "#",
         "# Regenerate: python reproducibility/build_lock.py",
         ""]
for i in sorted(items, key=lambda i: canonicalize_name(i["metadata"]["name"])):
    name, ver = i["metadata"]["name"], i["metadata"]["version"]
    ex = sorted(extras.get(canonicalize_name(name), ()))
    sha = i["download_info"]["archive_info"]["hashes"]["sha256"]
    if ex:  # old pip (e.g. 22.0) treats name[extra] and name as two requirements: list both
        lines.append(f"{name}[{','.join(ex)}]=={ver} \\\n    --hash=sha256:{sha}")
    lines.append(f"{name}=={ver} \\\n    --hash=sha256:{sha}")
LOCK.write_text("\n".join(lines) + "\n")
print(f"wrote {LOCK.relative_to(ROOT)}: {len(items)} packages, with extras: "
      f"{ {k: sorted(v) for k, v in extras.items() if v} }")
