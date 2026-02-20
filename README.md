# SQLite Chart Builder

Проект теперь поддерживает 2 интерфейса:

- `Web` (Dash + Plotly) - запуск в браузере, основной режим.
- `Desktop` (PySide6 + Plotly) - существующий Qt-вариант.

## Возможности веб-версии

- загрузка до 32 SQLite-файлов (`.db`, `.sqlite`, `.sqlite3`) прямо в браузере;
- ось `X` всегда `time@timestamp`;
- выбор диапазона времени и выбор конкретного дня;
- импорт до 32 файлов с объединением во временную общую базу;
- один график по объединенной базе с несколькими каналами;
- отдельная ось `Y` для каждого канала;
- фиксированные диапазоны по каналам;
- переключатель видимости осей `Y`;
- кнопки `Построить выбранные` и `Построить все`;
- кнопка `Сохранить график (PNG)` для скачивания файла на ПК;
- сводка значений по клику на график (время `X` + значения каналов);
- сохранение графика как картинки через кнопку камеры в панели Plotly.

## Запуск веб-версии (рекомендуется)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m app.web_app
```

После запуска откройте:

- `http://127.0.0.1:8050`

Импорт баз: загрузка файлов кнопкой/drag-and-drop (до 32 за раз).

Если хотите авто-открытие браузера:

```bash
DASH_OPEN_BROWSER=1 python -m app.web_app
```

Упрощенный запуск:

- macOS/Linux: `scripts/run_web.sh`
- Windows: `scripts\run_web.bat`

## Запуск desktop-версии (Qt)

```bash
python -m app.main
```

## Структура

- `app/web_app.py` - веб-приложение на Dash
- `app/main.py` - desktop-вход (PySide6)
- `app/ui/main_window.py` - desktop UI
- `app/data/sqlite_service.py` - доступ к SQLite
- `assets/style.css` - стили веб-интерфейса

## Сборка Windows EXE (web-вариант, без Python)

Сборка делает папку с `GraphicsWebApp.exe`, который запускает локальный веб-сервер и открывает браузер.

```powershell
cd Graphics
scripts\build_windows.bat
```
