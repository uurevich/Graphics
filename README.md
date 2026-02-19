# SQLite Chart Builder (PySide6 + Plotly)

Desktop app template for opening SQLite `.db` files and building charts from table data.

## Features

- Open SQLite database files (`.db`, `.sqlite`, `.sqlite3`)
- Fixed X axis: `time@timestamp` (Unix seconds)
- Time interval filter (`Start` / `End`)
- Quick day selection (`Use day`) and full-range reset
- Multi-channel selection for `data_format_0..data_format_7`
- Single chart with all selected channels (different color per channel)
- Separate Y-axis for each selected channel (individual scale and range)
- Build selected channels or all channels at once
- Export current chart to image (`.png`, `.jpg`, `.webp`, `.bmp`)
- Left sidebar for all controls and actions
- One large chart area on the right
- Row limit control for chart data
- Interactive controls from Plotly (zoom/pan via mouse + modebar) + `Reset zoom` button

## Project structure

- `app/main.py` - application entry point
- `app/ui/main_window.py` - main UI and chart rendering
- `app/data/sqlite_service.py` - SQLite schema/data access

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m app.main
```

## Build Windows EXE (without Python installed on target PC)

Build step must be run on Windows (or in Windows CI):

```powershell
cd Graphics
scripts\build_windows.bat
```

Result:

- `dist\GraphicsApp\GraphicsApp.exe`

This `.exe` can run on Windows machines without preinstalled Python.

Alternative (CI):

- Use GitHub Actions workflow: `.github/workflows/build-windows.yml`
- Run `Build Windows App` (manual `workflow_dispatch`) or push to `main`/`master`
- Download artifact `GraphicsApp-windows` (contains `GraphicsApp.exe`)

## Notes

- The app expects table `data` with column `time@timestamp`.
- X axis is shown as date/time labels based on Unix timestamps.
- Plot rendering is implemented via Plotly inside Qt WebEngine.
- If Plotly is missing, install dependencies again: `pip install -r requirements.txt`.
# Graphics
# Graphics
