import hashlib
import io
import json
import os
import platform
import shutil
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from keystroke.data import sha256_file
from keystroke.features import feature_schema
from keystroke.methods import make_method

MODEL_FILE = "model.joblib"
PROFILE_FILE = "profile.json"


class ProfileError(Exception):
    def __init__(self, error_code: str, message_ru: str) -> None:
        super().__init__(message_ru)
        self.error_code = error_code
        self.message_ru = message_ru


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    os.replace(tmp, path)


@dataclass(frozen=True)
class Profile:
    model: Any
    meta: dict[str, Any]

    @property
    def threshold(self) -> float:
        return float(self.meta["threshold"])

    def verify(self, x: np.ndarray) -> tuple[bool, float]:
        """Accept iff score <= threshold, same rule as the CMU experiment."""
        score = float(self.model.score(x.reshape(1, -1))[0])
        return score <= self.threshold, score


class ProfileStore:
    def __init__(self, root: Path, phrase: str, n_train: int, n_calib: int, percentile: float) -> None:
        self.root = root
        self.phrase = phrase
        self.schema = feature_schema(phrase)
        self.n_train, self.n_calib, self.percentile = n_train, n_calib, percentile
        self.pending_path = root / "pending" / "attempts.json"
        self.active_dir = root / "active"
        self.archive_dir = root / "archive"
        # Re-entrant: status() -> load_active(); one lock also serialises reads with os.replace on Windows.
        self._lock = threading.RLock()
        self._cached: Profile | None = None

    def _pending(self) -> dict[str, Any]:
        if not self.pending_path.is_file():
            return {"schema": self.schema["names"], "train": [], "calib": []}
        data = json.loads(self.pending_path.read_text(encoding="utf-8"))
        if data.get("schema") != self.schema["names"]:
            raise ProfileError("processing_error", "Незавершённый сбор сделан для другой схемы признаков. Сбросьте профиль.")
        return data

    def has_active(self) -> bool:
        return (self.active_dir / PROFILE_FILE).is_file()

    def status(self) -> dict[str, Any]:
        base = {"phrase": self.phrase, "n_train_needed": self.n_train, "n_calib_needed": self.n_calib}
        with self._lock:
            if self.has_active():
                p = self.load_active()
                return {**base, "status": "ready", "method": p.meta["method"], "threshold": p.threshold,
                        "created_utc": p.meta["created_utc"]}
            pending = self._pending()
        n_train, n_calib = len(pending["train"]), len(pending["calib"])
        return {**base, "status": "collecting" if n_train or n_calib else "none", "n_train": n_train, "n_calib": n_calib,
                "collected": n_train >= self.n_train and n_calib >= self.n_calib}

    def add_enrollment(self, x: np.ndarray) -> dict[str, Any]:
        with self._lock:
            if self.has_active():
                raise ProfileError("profile_exists", "Профиль уже создан. Пересоздание — только через сброс.")
            pending = self._pending()
            if len(pending["train"]) < self.n_train:
                role = "train"
            elif len(pending["calib"]) < self.n_calib:
                role = "calib"
            else:
                raise ProfileError("enrollment_complete", "Все вводы собраны — создайте профиль.")
            pending[role].append(x.tolist())
            _write_json(self.pending_path, pending)
            return {"role": role, "n_train": len(pending["train"]), "n_calib": len(pending["calib"])}

    def build(self, method: str, params: dict[str, Any]) -> Profile:
        with self._lock:
            if self.has_active():
                raise ProfileError("profile_exists", "Профиль уже создан.")
            pending = self._pending()
            if len(pending["train"]) < self.n_train or len(pending["calib"]) < self.n_calib:
                raise ProfileError(
                    "enrollment_incomplete",
                    f"Сбор не завершён: обучение {len(pending['train'])}/{self.n_train}, "
                    f"настройка порога {len(pending['calib'])}/{self.n_calib}.",
                )
            train = np.asarray(pending["train"][: self.n_train], dtype=np.float64)
            calib = np.asarray(pending["calib"][: self.n_calib], dtype=np.float64)
            model = make_method(method, params).fit(train)
            calib_scores = model.score(calib)
            threshold = float(np.percentile(calib_scores, self.percentile, method="linear"))

            self.active_dir.mkdir(parents=True, exist_ok=True)
            model_path = self.active_dir / MODEL_FILE
            tmp = model_path.with_suffix(".tmp")
            joblib.dump(model, tmp)
            os.replace(tmp, model_path)
            meta = {
                **self.schema,
                "method": method,
                "params": params,
                "threshold": threshold,
                "threshold_rule": f"{self.percentile:g}th percentile (linear) of {self.n_calib} held-out owner scores",
                "calib_scores": calib_scores.tolist(),
                "profile_mean": model.profile_mean().tolist(),
                "n_train": self.n_train,
                "n_calib": self.n_calib,
                "model_sha256": sha256_file(model_path),
                "created_utc": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
                "versions": {"python": platform.python_version(), "scikit-learn": version("scikit-learn"),
                             "numpy": version("numpy")},
            }
            _write_json(self.active_dir / PROFILE_FILE, meta)
            self.pending_path.unlink(missing_ok=True)
            self._cached = Profile(model, meta)
            return self._cached

    def load_active(self) -> Profile:
        with self._lock:
            if self._cached is not None and self.has_active():
                return self._cached
            self._cached = None
            meta_path, model_path = self.active_dir / PROFILE_FILE, self.active_dir / MODEL_FILE
            if not meta_path.is_file() or not model_path.is_file():
                raise ProfileError("no_profile", "Профиль не создан. Сначала соберите вводы и создайте профиль.")
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("names") != self.schema["names"] or meta.get("phrase") != self.phrase or meta.get("units") != "seconds":
                raise ProfileError("processing_error", "Схема признаков профиля не совпадает с текущей. Пересоздайте профиль.")
            # joblib is pickle and the file is local-only (never uploaded via the API). The hash sits next
            # to the model, so it detects corruption, not deliberate tampering; bytes are read once to avoid TOCTOU.
            data = model_path.read_bytes()
            if hashlib.sha256(data).hexdigest() != meta.get("model_sha256"):
                raise ProfileError("processing_error", "Файл модели профиля изменён или повреждён.")
            self._cached = Profile(joblib.load(io.BytesIO(data)), meta)
            return self._cached

    def reset(self) -> str | None:
        with self._lock:
            archived = None
            if self.active_dir.exists():
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                target = self.archive_dir / stamp
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(self.active_dir), str(target))
                archived = stamp
            self.pending_path.unlink(missing_ok=True)
            self._cached = None
            return archived
