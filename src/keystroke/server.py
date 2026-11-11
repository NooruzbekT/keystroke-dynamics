import math
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, StrictBool, field_validator
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from keystroke.config import ROOT, load_app_config, load_experiment_config, resolve
from keystroke.data import DataError
from keystroke.events import MAX_EVENTS, InvalidAttempt, RawEvent, attempt_duration, validate_attempt
from keystroke.experiment_log import CONDITIONS, append_record, counts_for_profile, read_records
from keystroke.features import compute_features, group_indices
from keystroke.frozen import load_frozen
from keystroke.profile_store import Profile, ProfileError, ProfileStore

MAX_BODY_BYTES = 64 * 1024
ALLOWED_HOSTS = ["127.0.0.1", "localhost"]
STATUS_CODES = {
    "no_profile": 404,
    "enrollment_incomplete": 409,
    "enrollment_complete": 409,
    "profile_exists": 409,
    "series_complete": 409,
    "wrong_phrase": 422,
    "invalid_attempt": 422,
    "processing_error": 500,
}


class KeyEvent(BaseModel):
    type: Literal["down", "up"]
    key: Annotated[str, Field(min_length=1, max_length=32)]
    code: Annotated[str, Field(min_length=1, max_length=32)]
    t: float

    @field_validator("t")
    @classmethod
    def finite(cls, v: float) -> float:
        if not math.isfinite(v) or v < 0:
            raise ValueError("t must be a finite non-negative number of milliseconds")
        return v


class Attempt(BaseModel):
    events: Annotated[list[KeyEvent], Field(min_length=1, max_length=MAX_EVENTS)]


class ExperimentAttempt(Attempt):
    condition: Literal["normal", "slow", "pause"]


class Confirmation(BaseModel):
    confirm: StrictBool = False


def error(code: str, message_ru: str) -> JSONResponse:
    return JSONResponse({"error_code": code, "message_ru": message_ru}, status_code=STATUS_CODES.get(code, 400))


class BodyLimitMiddleware:
    """Rejects non-GET/HEAD bodies over max_bytes with 413, by Content-Length or by streamed size (chunked included)."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] in ("GET", "HEAD"):
            await self.app(scope, receive, send)
            return
        too_large = JSONResponse({"error_code": "invalid_attempt", "message_ru": "Слишком большой запрос."}, status_code=413)
        length = dict(scope["headers"]).get(b"content-length")
        if length is not None and (not length.isdigit() or int(length) > self.max_bytes):
            await too_large(scope, receive, send)
            return
        chunks, size = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body = message.get("body", b"")
            size += len(body)
            if size > self.max_bytes:
                await too_large(scope, receive, send)
                return
            chunks.append(body)
            if not message.get("more_body", False):
                break
        replayed = False

        async def replay() -> Message:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": b"".join(chunks), "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


def create_app(
    profiles_dir: Path | None = None,
    logs_dir: Path | None = None,
    frozen_path: Path | None = None,
    web_dir: Path | None = None,
) -> FastAPI:
    app_cfg = load_app_config()
    phrase = app_cfg["phrase"]
    store = ProfileStore(
        profiles_dir or resolve(app_cfg["paths"]["profiles"]),
        phrase,
        int(app_cfg["n_train"]),
        int(app_cfg["n_calib"]),
        float(app_cfg["threshold_percentile"]),
    )
    log_path = (logs_dir or resolve(app_cfg["paths"]["logs"])) / "own_experiment.jsonl"
    frozen_path = frozen_path or resolve(load_experiment_config()["paths"]["frozen"])
    max_seconds = float(app_cfg["max_attempt_seconds"])
    per_condition = int(app_cfg["experiment_per_condition"])
    log_lock = threading.Lock()

    app = FastAPI(title="keystroke-dynamics", docs_url=None, redoc_url=None, openapi_url=None)
    # Local-only service: reject foreign Host headers (DNS rebinding).
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)
    app.add_middleware(BodyLimitMiddleware, max_bytes=MAX_BODY_BYTES)

    @app.exception_handler(RequestValidationError)
    async def on_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        return error("invalid_attempt", "Некорректный формат запроса.")

    @app.exception_handler(InvalidAttempt)
    async def on_invalid(request: Request, exc: InvalidAttempt) -> JSONResponse:
        return error(exc.error_code, exc.message_ru)

    @app.exception_handler(ProfileError)
    async def on_profile_error(request: Request, exc: ProfileError) -> JSONResponse:
        return error(exc.error_code, exc.message_ru)

    @app.exception_handler(Exception)
    async def on_unexpected(request: Request, exc: Exception) -> JSONResponse:
        return error("processing_error", "Ошибка обработки на сервере. Повторите попытку.")

    def features_of(attempt: Attempt):
        raw = [RawEvent(e.type, e.key, e.code, e.t) for e in attempt.events]
        strokes = validate_attempt(raw, phrase, max_seconds)
        return compute_features(strokes), attempt_duration(strokes)

    def active_profile() -> Profile:
        status = store.status()
        if status["status"] == "collecting":
            message = (
                "Все вводы собраны — нажмите «Создать профиль»."
                if status["collected"]
                else "Сбор вводов не завершён — сначала соберите вводы и создайте профиль."
            )
            raise ProfileError("enrollment_incomplete", message)
        return store.load_active()

    def verdict(profile: Profile, attempt: Attempt) -> dict[str, Any]:
        x, duration = features_of(attempt)
        accepted, score = profile.verify(x)
        names = profile.meta["names"]
        return {
            "accepted": accepted,
            "score": score,
            "threshold": profile.threshold,
            "duration": duration,
            "features": {"names": names, "values": x.tolist()},
            "profile_mean": profile.meta["profile_mean"],
            "groups": {g: group_indices(names, f"{g}.") for g in ("H", "DD", "UD")},
        }

    def experiment_counts(profile: Profile) -> dict[str, int]:
        records, _ = read_records(log_path)
        return counts_for_profile(records, profile.meta["created_utc"])

    @app.get("/api/profile")
    def get_profile() -> dict[str, Any]:
        return {**store.status(), "max_attempt_seconds": max_seconds, "experiment_per_condition": per_condition}

    @app.post("/api/enroll/attempt")
    def enroll_attempt(attempt: Attempt) -> dict[str, Any]:
        if store.has_active():
            raise ProfileError("profile_exists", "Профиль уже создан. Пересоздание — только через сброс.")
        x, duration = features_of(attempt)
        return {**store.add_enrollment(x), "duration": duration}

    @app.post("/api/profile/build")
    def build_profile(req: Confirmation) -> dict[str, Any]:
        if not req.confirm:
            raise InvalidAttempt("invalid_attempt", "Создание профиля требует подтверждения.")
        try:
            app_method = load_frozen(frozen_path)["app_method"]
        except (DataError, KeyError) as exc:
            raise ProfileError("processing_error", "Нет config/frozen.json с выбранным методом (scripts/tune.py).") from exc
        store.build(app_method["name"], app_method["params"])
        return store.status()

    @app.post("/api/profile/reset")
    def reset_profile(req: Confirmation) -> dict[str, Any]:
        if not req.confirm:
            raise InvalidAttempt("invalid_attempt", "Сброс требует подтверждения.")
        return {"archived": store.reset(), **store.status()}

    @app.post("/api/verify")
    def verify(attempt: Attempt) -> dict[str, Any]:
        return verdict(active_profile(), attempt)

    @app.get("/api/experiment")
    def experiment_status() -> dict[str, Any]:
        counts = experiment_counts(store.load_active()) if store.has_active() else dict.fromkeys(CONDITIONS, 0)
        return {"counts": counts, "per_condition": per_condition}

    @app.post("/api/experiment/attempt")
    def experiment_attempt(attempt: ExperimentAttempt) -> dict[str, Any]:
        profile = active_profile()
        result = verdict(profile, attempt)
        record = {
            "ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "condition": attempt.condition,
            "accepted": result["accepted"],
            "score": result["score"],
            "threshold": result["threshold"],
            "duration": result["duration"],
            "profile_created_utc": profile.meta["created_utc"],
            "features": result["features"]["values"],
        }
        with log_lock:
            if experiment_counts(profile)[attempt.condition] >= per_condition:
                raise ProfileError("series_complete", f"В этой серии уже {per_condition} попыток. Выберите другую серию.")
            append_record(log_path, record)
            counts = experiment_counts(profile)
        return {**result, "condition": attempt.condition, "counts": counts, "per_condition": per_condition}

    app.frontend("/", directory=str(web_dir or ROOT / "web"))
    return app
