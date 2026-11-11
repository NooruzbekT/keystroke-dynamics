from dataclasses import dataclass
from typing import Literal

MAX_EVENTS = 200
IGNORED_KEYS = {"Enter"}
FORBIDDEN_KEYS = {
    "Backspace": "Использован Backspace — исправления запрещены.",
    "Delete": "Использован Delete — исправления запрещены.",
    "Shift": "Нажат Shift — фраза вводится строчными буквами без модификаторов.",
    "Control": "Нажат Ctrl — модификаторы запрещены.",
    "Alt": "Нажат Alt — модификаторы запрещены.",
    "AltGraph": "Нажат AltGr — модификаторы запрещены.",
    "Meta": "Нажата клавиша Win/Cmd — модификаторы запрещены.",
    "CapsLock": "Нажат CapsLock — фраза вводится строчными буквами.",
    "Tab": "Нажат Tab — ввод прерван.",
    "Escape": "Нажат Esc — ввод прерван.",
    "ArrowLeft": "Курсор перемещён стрелкой — ввод должен идти подряд.",
    "ArrowRight": "Курсор перемещён стрелкой — ввод должен идти подряд.",
    "ArrowUp": "Курсор перемещён стрелкой — ввод должен идти подряд.",
    "ArrowDown": "Курсор перемещён стрелкой — ввод должен идти подряд.",
    "Home": "Нажат Home — ввод должен идти подряд.",
    "End": "Нажат End — ввод должен идти подряд.",
}


class InvalidAttempt(ValueError):
    def __init__(self, error_code: str, message_ru: str) -> None:
        super().__init__(message_ru)
        self.error_code = error_code
        self.message_ru = message_ru


@dataclass(frozen=True)
class RawEvent:
    type: Literal["down", "up"]
    key: str
    code: str
    t_ms: float


@dataclass(frozen=True)
class Keystroke:
    char: str
    code: str
    down: float
    up: float


def validate_attempt(events: list[RawEvent], phrase: str, max_seconds: float) -> list[Keystroke]:
    """Replay browser events; any deviation rejects the whole attempt. Times are converted ms -> s."""
    if len(events) > MAX_EVENTS:
        raise InvalidAttempt("invalid_attempt", "Слишком много событий в попытке.")
    typed: list[dict] = []
    pressed: dict[str, int] = {}
    last_t = float("-inf")

    for e in events:
        if e.t_ms < last_t:
            raise InvalidAttempt("invalid_attempt", "Нарушен порядок событий по времени.")
        last_t = e.t_ms
        if e.key in IGNORED_KEYS:
            continue
        if e.key in FORBIDDEN_KEYS:
            raise InvalidAttempt("invalid_attempt", FORBIDDEN_KEYS[e.key])

        if e.type == "down":
            if e.code in pressed:
                raise InvalidAttempt("invalid_attempt", "Обнаружен автоповтор удерживаемой клавиши.")
            if len(typed) >= len(phrase):
                raise InvalidAttempt("wrong_phrase", "Введён лишний символ после конца фразы.")
            expected = phrase[len(typed)]
            if e.key != expected:
                raise InvalidAttempt(
                    "wrong_phrase",
                    f"Ожидался символ «{expected}», введён «{e.key}». Проверьте фразу и раскладку.",
                )
            pressed[e.code] = len(typed)
            typed.append({"char": e.key, "code": e.code, "down": e.t_ms, "up": None})
        else:
            idx = pressed.pop(e.code, None)
            if idx is None:
                raise InvalidAttempt("invalid_attempt", "Отпускание клавиши без нажатия — попытка повреждена.")
            typed[idx]["up"] = e.t_ms

    if len(typed) < len(phrase):
        raise InvalidAttempt("wrong_phrase", "Фраза введена не полностью.")
    if pressed:
        raise InvalidAttempt("invalid_attempt", "Не все клавиши отпущены — попытка не завершена.")

    strokes = [Keystroke(k["char"], k["code"], k["down"] / 1000.0, k["up"] / 1000.0) for k in typed]
    if attempt_duration(strokes) > max_seconds:
        raise InvalidAttempt("invalid_attempt", f"Попытка длилась дольше {max_seconds:g} с.")
    return strokes


def attempt_duration(strokes: list[Keystroke]) -> float:
    return max(k.up for k in strokes) - strokes[0].down
