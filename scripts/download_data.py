"""Fetch the CMU Keystroke Dynamics Benchmark CSV if missing, validate it, and record or verify its SHA-256."""

import http.client
import re
import sys
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

from keystroke.config import EXPERIMENT_CONFIG, ROOT, load_experiment_config, resolve, utf8_stdio
from keystroke.data import DataError, read_csv, sha256_file, validate_schema, verify_sha256

DATA_README = ROOT / "data" / "README.md"
MAX_BYTES = 50 * 1024 * 1024


class DownloadError(RuntimeError):
    pass


def manual_instructions(url: str, path: Path) -> str:
    return (
        "Не удалось скачать датасет автоматически.\n"
        f"1. Скачайте файл вручную: {url}\n"
        f"2. Сохраните его без изменений как: {path}\n"
        "3. Запустите скрипт ещё раз: python scripts/download_data.py"
    )


def download(url: str, path: Path) -> None:
    if urlparse(url).scheme != "https":
        raise DownloadError(f"only https URLs are allowed: {url}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "keystroke-dynamics-downloader"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp, tmp.open("wb") as out:
            # urllib follows redirects to plain http; reject them.
            if urlparse(resp.geturl()).scheme != "https":
                raise DownloadError(f"redirected to a non-https URL: {resp.geturl()}")
            total = 0
            while chunk := resp.read(1 << 16):
                total += len(chunk)
                if total > MAX_BYTES:
                    raise DownloadError(f"response exceeds {MAX_BYTES} bytes")
                out.write(chunk)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def record_hash(sha: str, retrieved: str) -> None:
    def sub_once(pattern: str, repl: str, text: str, where: str) -> str:
        text, n = re.subn(pattern, repl, text, count=1, flags=re.M)
        if n != 1:
            raise DataError(f"{where}: line matching {pattern!r} not found")
        return text

    text = EXPERIMENT_CONFIG.read_text(encoding="utf-8")
    text = sub_once(r"^(\s*)sha256:.*$", rf"\g<1>sha256: {sha}", text, "config/experiment.yaml")
    text = sub_once(r"^(\s*)retrieved:.*$", rf'\g<1>retrieved: "{retrieved}"', text, "config/experiment.yaml")

    readme = DATA_README.read_text(encoding="utf-8")
    readme = sub_once(r"^- \*\*SHA-256:\*\*.*$", f"- **SHA-256:** `{sha}`", readme, "data/README.md")
    readme = sub_once(r"^- \*\*Дата получения:\*\*.*$", f"- **Дата получения:** {retrieved}", readme, "data/README.md")

    EXPERIMENT_CONFIG.write_text(text, encoding="utf-8", newline="\n")
    DATA_README.write_text(readme, encoding="utf-8", newline="\n")


def main() -> int:
    utf8_stdio()
    cfg = load_experiment_config()["data"]
    url, path, recorded = cfg["url"], resolve(cfg["path"]), cfg.get("sha256")

    if not path.exists():
        print(f"Downloading {url}")
        try:
            download(url, path)
        except (urllib.error.URLError, http.client.HTTPException, TimeoutError, OSError, DownloadError) as exc:
            print(f"Download failed: {exc}\n", file=sys.stderr)
            print(manual_instructions(url, path), file=sys.stderr)
            return 1
    else:
        print(f"Found existing file: {path}")

    try:
        sha = verify_sha256(path, recorded) if recorded else sha256_file(path)
        features = validate_schema(read_csv(path))
    except DataError as exc:
        print(exc, file=sys.stderr)
        print(f"Удалите {path} и скачайте заново, либо выясните, почему файл отличается.", file=sys.stderr)
        return 1

    if not recorded:
        record_hash(sha, date.today().isoformat())
        print("Recorded SHA-256 in config/experiment.yaml and data/README.md")

    print(f"OK: {path.name}, sha256={sha}, {len(features)} feature columns")
    return 0


if __name__ == "__main__":
    sys.exit(main())
