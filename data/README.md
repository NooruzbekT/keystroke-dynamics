# Данные: CMU Keystroke Dynamics Benchmark

- **Источник:** Keystroke Dynamics — Benchmark Data Set, Carnegie Mellon University, страница авторов: https://www.cs.cmu.edu/~keystroke/
- **Файл:** `DSL-StrongPasswordData.csv` (https://www.cs.cmu.edu/~keystroke/DSL-StrongPasswordData.csv)
- **Способ загрузки:** `python scripts/download_data.py` — скачивает файл в `data/raw/`, считает SHA-256, при первом запуске записывает хэш в `config/experiment.yaml` и сюда, при следующих — сверяет и падает при расхождении.
- **SHA-256:** `b11d23538b1865fa6ecf4e8b78567caa312e9c1027604bb022fcc6ad7eaa7a33`
- **Дата получения:** 2026-11-09

Сам CSV в репозиторий не коммитится (`data/raw/` в `.gitignore`).

## Если сеть недоступна

Скачайте файл по ссылке выше вручную и положите его без изменений в `data/raw/DSL-StrongPasswordData.csv`, затем запустите `python scripts/download_data.py` — он сверит хэш со значением выше и проверит схему.

## Состав

51 участник × 8 сессий × 50 повторений = 20 400 строк. Каждый участник вводил пароль `.tie5Roanl` и Enter.

- Служебные столбцы: `subject` (`s002`…), `sessionIndex` (1–8), `rep` (1–50). Используются только для формирования выборок, во вход моделей не подаются.
- 31 временной признак в секундах: 11 `H.<клавиша>` (удержание), 10 `DD.<a>.<b>` (keydown→keydown), 10 `UD.<a>.<b>` (keyup→keydown). Отрицательный `UD` означает перекрытие нажатий и является нормальным значением.

## Цитирование

Авторы просят ссылаться на статью:

> Kevin S. Killourhy, Roy A. Maxion. *Comparing Anomaly Detectors for Keystroke Dynamics.* In Proceedings of the 39th Annual International Conference on Dependable Systems and Networks (DSN-2009), pages 125–134, Estoril, Lisbon, Portugal, June 2009. IEEE Computer Society Press.
