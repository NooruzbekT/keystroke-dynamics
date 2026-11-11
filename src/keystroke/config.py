import platform
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_CONFIG = ROOT / "config" / "experiment.yaml"
APP_CONFIG = ROOT / "config" / "app.yaml"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_experiment_config() -> dict[str, Any]:
    return load_yaml(EXPERIMENT_CONFIG)


def load_app_config() -> dict[str, Any]:
    return load_yaml(APP_CONFIG)


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def utf8_stdio() -> None:
    """Redirected stdout/stderr on Windows default to a legacy codepage that cannot encode Cyrillic."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def git_state() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True)
        except (OSError, subprocess.CalledProcessError):
            return None
        return out.stdout.strip()

    # Untracked outputs (frozen.json itself, reports) do not make the code revision ambiguous.
    status = run("status", "--porcelain", "--untracked-files=no")
    return {"git_commit": run("rev-parse", "HEAD"), "git_dirty": bool(status) if status is not None else None}


def library_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        **{pkg: version(pkg) for pkg in ("numpy", "pandas", "scikit-learn")},
    }
