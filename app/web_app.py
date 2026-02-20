import base64
import math
import os
import sqlite3
import tempfile
import webbrowser
from bisect import bisect_left
from datetime import datetime, timedelta
from pathlib import Path
from threading import Timer
from typing import Any, Optional

import plotly.graph_objects as go
try:
    from dash import Dash, Input, Output, State, callback, ctx, dcc, html, no_update
    from dash.exceptions import PreventUpdate
except ImportError as error:  # pragma: no cover - runtime dependency guard
    raise SystemExit(
        "Не найден пакет 'dash'. Установите зависимости: pip install -r requirements.txt"
    ) from error

from app.data.sqlite_service import SQLiteService


DATA_TABLE = "data"
TIME_COLUMN = "time@timestamp"
MAX_RENDER_POINTS_PER_CHANNEL = 3500
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = PROJECT_ROOT / "assets"

Y_CHANNELS = [
    ("data_format_0", "Объект мойки (INT16)"),
    ("data_format_1", "Программа мойки (INT16)"),
    ("data_format_2", "Процесс (INT16)"),
    ("data_format_3", "Расход подачи (REAL32)"),
    ("data_format_4", "Давление подачи (REAL32)"),
    ("data_format_5", "Температура подачи (REAL32)"),
    ("data_format_6", "Температура возврата (REAL32)"),
    ("data_format_7", "Концентрация возврата (REAL32)"),
]

SERIES_COLORS = [
    "#D62828",
    "#1D3557",
    "#0B8A7A",
    "#C97A14",
    "#7A3EB1",
    "#2A6F97",
    "#B33951",
    "#2F4858",
]

FIXED_Y_RANGES: dict[str, tuple[float, float]] = {
    "data_format_0": (0.0, 50.0),
    "data_format_1": (0.0, 20.0),
    "data_format_2": (0.0, 50.0),
    "data_format_3": (0.0, 50.0),
    "data_format_4": (0.0, 20.0),
    "data_format_5": (0.0, 100.0),
    "data_format_6": (0.0, 100.0),
    "data_format_7": (0.0, 5.0),
}


def _to_float_or_none(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalized_numeric_range(min_value: float, max_value: float) -> tuple[float, float]:
    if min_value == max_value:
        pad = 1.0 if min_value == 0 else abs(min_value) * 0.05
        return min_value - pad, max_value + pad
    return min_value, max_value


def _downsample_lttb(
    x_values: list[float],
    y_values: list[float],
    threshold: int,
) -> tuple[list[float], list[float]]:
    data_len = len(x_values)
    if data_len == 0:
        return [], []
    if threshold >= data_len or threshold < 3:
        return x_values, y_values

    sampled_x = [x_values[0]]
    sampled_y = [y_values[0]]
    every = (data_len - 2) / (threshold - 2)
    a = 0

    for i in range(threshold - 2):
        avg_range_start = int(math.floor((i + 1) * every)) + 1
        avg_range_end = int(math.floor((i + 2) * every)) + 1
        if avg_range_end >= data_len:
            avg_range_end = data_len
        if avg_range_start >= data_len:
            avg_range_start = data_len - 1

        avg_range_length = max(1, avg_range_end - avg_range_start)
        avg_x = sum(x_values[avg_range_start:avg_range_end]) / avg_range_length
        avg_y = sum(y_values[avg_range_start:avg_range_end]) / avg_range_length

        range_offs = int(math.floor(i * every)) + 1
        range_to = int(math.floor((i + 1) * every)) + 1
        if range_to >= data_len:
            range_to = data_len - 1

        point_a_x = x_values[a]
        point_a_y = y_values[a]
        max_area = -1.0
        next_a = range_offs

        for idx in range(range_offs, max(range_offs + 1, range_to)):
            area = abs(
                (point_a_x - avg_x) * (y_values[idx] - point_a_y)
                - (point_a_x - x_values[idx]) * (avg_y - point_a_y)
            )
            if area > max_area:
                max_area = area
                next_a = idx

        sampled_x.append(x_values[next_a])
        sampled_y.append(y_values[next_a])
        a = next_a

    sampled_x.append(x_values[-1])
    sampled_y.append(y_values[-1])
    return sampled_x, sampled_y


def _format_number(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _strip_type_suffix(label: str) -> str:
    text = (label or "").strip()
    if text.endswith(")") and " (" in text:
        return text.rsplit(" (", 1)[0]
    return text


def _parse_plotly_x_to_timestamp(x_value: str) -> Optional[float]:
    text = (x_value or "").strip()
    if not text:
        return None

    try:
        numeric = float(text)
        if numeric > 100000000000:
            return numeric / 1000.0
        return numeric
    except ValueError:
        pass

    normalized = text.replace("T", " ").replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized).timestamp()
    except ValueError:
        pass

    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(normalized, fmt).timestamp()
        except ValueError:
            continue
    return None


def _timestamp_to_input_value(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%dT%H:%M:%S")


def _input_value_to_timestamp(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return None


def _build_day_options(min_ts: float, max_ts: float) -> list[dict[str, str]]:
    start_date = datetime.fromtimestamp(min_ts).date()
    end_date = datetime.fromtimestamp(max_ts).date()
    days = (end_date - start_date).days
    options: list[dict[str, str]] = []
    for offset in range(days + 1):
        day = start_date + timedelta(days=offset)
        options.append({"label": day.strftime("%d.%m.%Y"), "value": day.isoformat()})
    return options


def _build_meta_line(min_ts: float, max_ts: float, channel_count: int) -> str:
    start_text = datetime.fromtimestamp(min_ts).strftime("%d.%m.%Y %H:%M:%S")
    end_text = datetime.fromtimestamp(max_ts).strftime("%d.%m.%Y %H:%M:%S")
    return f"Диапазон: {start_text} - {end_text} | Каналов: {channel_count}"


def _cleanup_temp_db(path: Optional[str]) -> None:
    if not path:
        return
    candidate = Path(path)
    if not candidate.exists():
        return
    if not candidate.name.startswith("dash_chart_"):
        return
    try:
        candidate.unlink()
    except OSError:
        return


def _decode_upload_to_temp_db(contents: str) -> str:
    if "," not in contents:
        raise ValueError("Некорректный формат загруженного файла.")
    encoded = contents.split(",", 1)[1]
    data = base64.b64decode(encoded)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        suffix=".db",
        prefix="dash_chart_",
        delete=False,
    ) as temp_file:
        temp_file.write(data)
        return temp_file.name


def _validate_input_path(path_text: str) -> str:
    raw = (path_text or "").strip().strip("\"'")
    if not raw:
        raise ValueError("Введите путь к файлу базы данных.")
    candidate = Path(raw).expanduser()
    if not candidate.exists():
        raise ValueError("Файл по указанному пути не найден.")
    if not candidate.is_file():
        raise ValueError("Указанный путь не является файлом.")
    return str(candidate.resolve())


def _current_triggered_id() -> Optional[str]:
    try:
        return ctx.triggered_id
    except Exception:
        return None


def _build_placeholder_figure(title: str) -> go.Figure:
    figure = go.Figure()
    figure.update_layout(
        title=title,
        template="plotly_white",
        margin={"l": 40, "r": 40, "t": 70, "b": 60},
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[
            {
                "text": "Нет данных для отображения",
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.5,
                "showarrow": False,
                "font": {"size": 16, "color": "#475569"},
            }
        ],
    )
    return figure


def _plotly_config() -> dict[str, object]:
    return {
        "responsive": True,
        "displaylogo": False,
        "toImageButtonOptions": {"format": "png", "filename": "chart"},
    }


def _build_traces_payload(
    service: SQLiteService,
    selected_channels: list[tuple[str, str]],
    limit: int,
    start_ts: float,
    end_ts: float,
) -> tuple[list[dict[str, object]], int]:
    channel_columns = [column for column, _ in selected_channels]
    rows = service.fetch_multi_series_rows(
        table_name=DATA_TABLE,
        y_columns=channel_columns,
        limit=limit,
        start_ts=start_ts,
        end_ts=end_ts,
    )

    buffers: dict[str, dict[str, object]] = {}
    for index, (column_name, label) in enumerate(selected_channels):
        buffers[column_name] = {
            "name": label,
            "color": SERIES_COLORS[index % len(SERIES_COLORS)],
            "ts_values": [],
            "y_values": [],
            "min_y": float("inf"),
            "max_y": float("-inf"),
        }

    for row in rows:
        ts_seconds = _to_float_or_none(row["time_value"])
        if ts_seconds is None:
            continue
        for column_name, _ in selected_channels:
            y_value = _to_float_or_none(row[column_name])
            if y_value is None:
                continue
            buffer = buffers[column_name]
            buffer["ts_values"].append(ts_seconds)  # type: ignore[index]
            buffer["y_values"].append(y_value)      # type: ignore[index]
            buffer["min_y"] = min(buffer["min_y"], y_value)  # type: ignore[index]
            buffer["max_y"] = max(buffer["max_y"], y_value)  # type: ignore[index]

    downsample_target = max(500, min(limit, MAX_RENDER_POINTS_PER_CHANNEL))
    traces_payload: list[dict[str, object]] = []
    points_total = 0

    for column_name, label in selected_channels:
        buffer = buffers[column_name]
        ts_values: list[float] = buffer["ts_values"]  # type: ignore[assignment]
        y_values: list[float] = buffer["y_values"]    # type: ignore[assignment]
        if not ts_values:
            continue

        sampled_ts, sampled_y = _downsample_lttb(ts_values, y_values, downsample_target)

        fixed_range = FIXED_Y_RANGES.get(column_name)
        if fixed_range is None:
            min_y = float(buffer["min_y"])  # type: ignore[arg-type]
            max_y = float(buffer["max_y"])  # type: ignore[arg-type]
            y_min, y_max = _normalized_numeric_range(min_y, max_y)
        else:
            y_min, y_max = fixed_range

        traces_payload.append(
            {
                "column": column_name,
                "name": label,
                "color": buffer["color"],
                "ts_values": sampled_ts,
                "y_values": sampled_y,
                "y_min": y_min,
                "y_max": y_max,
            }
        )
        points_total += len(sampled_ts)

    return traces_payload, points_total


def _build_plotly_figure(
    traces_payload: list[dict[str, object]],
    start_ts: float,
    end_ts: float,
    show_y_axes: bool,
) -> go.Figure:
    figure = go.Figure()

    for index, payload in enumerate(traces_payload):
        yaxis_name = "y" if index == 0 else f"y{index + 1}"
        x_values = [datetime.fromtimestamp(ts) for ts in payload["ts_values"]]  # type: ignore[index]
        figure.add_trace(
            go.Scatter(
                x=x_values,
                y=payload["y_values"],  # type: ignore[index]
                mode="lines",
                name=payload["name"],  # type: ignore[index]
                line={"color": payload["color"], "width": 2},  # type: ignore[index]
                yaxis=yaxis_name,
                hoverinfo="skip",
                hovertemplate=None,
            )
        )

    if show_y_axes:
        left_extra = [idx for idx in range(1, len(traces_payload)) if idx % 2 == 0]
        right_extra = [idx for idx in range(1, len(traces_payload)) if idx % 2 == 1]

        left_pad = min(0.08 + 0.055 * len(left_extra), 0.36)
        right_pad = min(0.06 + 0.055 * len(right_extra), 0.36)
        if left_pad + right_pad > 0.72:
            scale = 0.72 / (left_pad + right_pad)
            left_pad *= scale
            right_pad *= scale
        x_domain = [left_pad, 1.0 - right_pad]

        left_positions: dict[int, float] = {}
        right_positions: dict[int, float] = {}
        for order, idx in enumerate(left_extra, start=1):
            left_positions[idx] = max(0.0, x_domain[0] - 0.055 * order)
        for order, idx in enumerate(right_extra, start=1):
            right_positions[idx] = min(1.0, x_domain[1] + 0.055 * order)
    else:
        x_domain = [0.04, 0.995]
        left_positions = {}
        right_positions = {}

    layout_axes = {}
    for index, payload in enumerate(traces_payload):
        axis_key = "yaxis" if index == 0 else f"yaxis{index + 1}"
        base_cfg = {
            "range": [payload["y_min"], payload["y_max"]],  # type: ignore[index]
            "zeroline": False,
        }
        if show_y_axes:
            base_cfg.update(
                {
                    "title": {"text": payload["name"], "font": {"color": payload["color"]}},  # type: ignore[index]
                    "tickfont": {"color": payload["color"]},  # type: ignore[index]
                }
            )
        else:
            base_cfg.update({"visible": False, "showgrid": False, "showticklabels": False})

        if index == 0:
            base_cfg.update({"showgrid": show_y_axes})
        else:
            side = "left" if index in left_positions else "right"
            position = left_positions.get(index, right_positions.get(index, x_domain[1]))
            base_cfg.update(
                {
                    "showgrid": False,
                    "overlaying": "y",
                    "anchor": "free",
                    "side": side,
                    "position": position,
                }
            )
        layout_axes[axis_key] = base_cfg

    start_dt = datetime.fromtimestamp(start_ts)
    end_dt = datetime.fromtimestamp(end_ts)
    figure.update_layout(
        title=(
            f"Каналы: {len(traces_payload)} | "
            f"{start_dt.strftime('%d.%m.%Y %H:%M:%S')} - "
            f"{end_dt.strftime('%d.%m.%Y %H:%M:%S')}"
        ),
        xaxis={
            "title": {"text": "Время"},
            "type": "date",
            "domain": x_domain,
            "showgrid": True,
            "rangeslider": {"visible": False},
            "showspikes": False,
        },
        hovermode=False,
        hoverdistance=0,
        spikedistance=0,
        legend={"orientation": "h", "y": -0.2, "x": 0.0},
        margin={"l": 45, "r": 45, "t": 70, "b": 100},
        template="plotly_white",
        **layout_axes,
    )
    return figure


def _db_channel_labels() -> dict[str, str]:
    return {column: label for column, label in Y_CHANNELS}


app = Dash(
    __name__,
    title="Построение графиков",
    assets_folder=str(ASSETS_DIR),
    assets_url_path="/assets",
)
server = app.server

app.layout = html.Div(
    className="app-shell",
    children=[
        dcc.Store(id="db-store"),
        dcc.Store(id="traces-store"),
        html.Aside(
            className="left-panel",
            children=[
                html.Div(
                    className="panel-head",
                    children=[
                        html.H1("Построение графиков", className="app-title"),
                        html.P(
                            "Работа с SQLite-файлами в браузере. Все настройки слева, график справа.",
                            className="subtitle",
                        ),
                        html.Span("Веб-интерфейс", className="pill"),
                    ],
                ),
                html.Div(
                    className="panel-card",
                    children=[
                        dcc.Upload(
                            id="db-upload",
                            className="upload-area",
                            children=html.Div(
                                [
                                    html.Span("Перетащите .db сюда или "),
                                    html.Span("выберите файл", className="linkish"),
                                ]
                            ),
                            multiple=False,
                            accept=".db,.sqlite,.sqlite3,application/octet-stream",
                        ),
                        html.Div("или откройте файл по пути", className="mini-label"),
                        html.Div(
                            className="row",
                            children=[
                                dcc.Input(
                                    id="db-path-input",
                                    type="text",
                                    className="text-input grow",
                                    placeholder="Например: C:\\data\\20250506_Canal#1.db",
                                ),
                                html.Button("Открыть", id="open-path-btn", className="ghost-btn"),
                            ],
                        ),
                        html.Div("Файл не выбран", id="db-info", className="db-info"),
                        html.Div("Диапазон: — | Каналов: —", id="meta-line", className="meta-line"),
                    ],
                ),
                html.Div("Параметры построения", className="section-title"),
                html.Label("Начало", className="input-label"),
                dcc.Input(id="start-dt", type="datetime-local", className="text-input"),
                html.Label("Конец", className="input-label"),
                dcc.Input(id="end-dt", type="datetime-local", className="text-input"),
                html.Div(
                    className="row",
                    children=[
                        html.Div(
                            className="grow",
                            children=[
                                html.Label("День", className="input-label"),
                                dcc.Dropdown(
                                    id="day-dropdown",
                                    options=[],
                                    value=None,
                                    placeholder="Выберите день",
                                    className="control-dropdown",
                                    clearable=False,
                                ),
                            ],
                        ),
                        html.Button("Выбрать день", id="apply-day-btn", className="ghost-btn"),
                    ],
                ),
                html.Button("Полный диапазон", id="full-range-btn", className="ghost-btn wide-btn"),
                html.Label("Строк на канал", className="input-label"),
                dcc.Input(
                    id="limit-input",
                    type="number",
                    value=4511,
                    min=10,
                    max=1_000_000,
                    step=100,
                    className="text-input",
                ),
                html.Div("Каналы Y", className="section-title"),
                dcc.Checklist(
                    id="channels-checklist",
                    options=[],
                    value=[],
                    className="channels-checklist",
                ),
                html.Div(
                    className="row",
                    children=[
                        html.Button("Выбрать все", id="select-all-btn", className="ghost-btn"),
                        html.Button("Очистить", id="clear-btn", className="ghost-btn"),
                    ],
                ),
                dcc.Checklist(
                    id="show-y-axes",
                    options=[{"label": "Показывать оси Y", "value": "show"}],
                    value=["show"],
                    className="single-check",
                ),
                html.Div(
                    className="row",
                    children=[
                        html.Button("Построить выбранные", id="build-selected-btn", className="accent-btn"),
                        html.Button("Построить все", id="build-all-btn", className="accent-btn secondary"),
                    ],
                ),
                html.Div(
                    "Сохранение картинки: кнопка камеры на панели графика.",
                    className="hint",
                ),
            ],
        ),
        html.Main(
            className="right-panel",
            children=[
                html.Div(
                    className="right-toolbar",
                    children=[
                        html.Div("График параметров", className="toolbar-title"),
                        html.Div(id="status-line", className="status-line"),
                    ],
                ),
                dcc.Loading(
                    type="dot",
                    className="graph-loading",
                    children=[
                        dcc.Graph(
                            id="main-chart",
                            figure=_build_placeholder_figure("Откройте файл базы данных и постройте график."),
                            config=_plotly_config(),
                            className="main-chart",
                        )
                    ],
                ),
                html.Div(
                    className="inspector-card",
                    children=[
                        html.Div("Значения по клику", className="inspector-title"),
                        html.Div("Время X: —", id="clicked-time", className="clicked-time"),
                        html.Ul(id="clicked-values", className="clicked-values"),
                    ],
                ),
            ],
        ),
    ],
)


@callback(
    Output("db-store", "data"),
    Output("db-info", "children"),
    Output("meta-line", "children"),
    Output("channels-checklist", "options"),
    Output("channels-checklist", "value"),
    Output("start-dt", "value"),
    Output("end-dt", "value"),
    Output("start-dt", "min"),
    Output("start-dt", "max"),
    Output("end-dt", "min"),
    Output("end-dt", "max"),
    Output("day-dropdown", "options"),
    Output("day-dropdown", "value"),
    Input("db-upload", "contents"),
    Input("open-path-btn", "n_clicks"),
    State("db-upload", "filename"),
    State("db-path-input", "value"),
    State("db-store", "data"),
    prevent_initial_call=True,
)
def load_database(
    contents: Optional[str],
    _open_path_clicks: Optional[int],
    filename: Optional[str],
    path_input: Optional[str],
    db_store: Optional[dict[str, Any]],
):
    previous_path = None
    if isinstance(db_store, dict):
        previous_path = db_store.get("db_path")

    temp_db_path: Optional[str] = None
    db_path_to_use: Optional[str] = None
    display_name: Optional[str] = None
    try:
        triggered = _current_triggered_id()
        if triggered == "open-path-btn":
            db_path_to_use = _validate_input_path(path_input or "")
            display_name = db_path_to_use
        else:
            if not contents:
                raise PreventUpdate
            temp_db_path = _decode_upload_to_temp_db(contents)
            db_path_to_use = temp_db_path
            display_name = filename or Path(temp_db_path).name

        assert db_path_to_use is not None
        service = SQLiteService(db_path_to_use)

        if not service.table_exists(DATA_TABLE):
            raise ValueError(f"Таблица '{DATA_TABLE}' не найдена.")

        columns = service.list_columns(DATA_TABLE)
        column_names = {column.name for column in columns}
        if TIME_COLUMN not in column_names:
            raise ValueError(f"В таблице '{DATA_TABLE}' нет столбца '{TIME_COLUMN}'.")

        available_channels = [(column, label) for column, label in Y_CHANNELS if column in column_names]
        if not available_channels:
            raise ValueError("В таблице не найдены поддерживаемые каналы Y.")

        bounds = service.fetch_time_bounds(DATA_TABLE, TIME_COLUMN)
        if not bounds:
            raise ValueError("Не удалось определить временной диапазон в таблице данных.")

        min_ts, max_ts = bounds
        try:
            service.ensure_time_index(DATA_TABLE, TIME_COLUMN)
        except sqlite3.Error:
            pass

        _cleanup_temp_db(previous_path)

        options = [{"label": label, "value": column} for column, label in available_channels]
        default_selection = [available_channels[0][0]]
        start_value = _timestamp_to_input_value(min_ts)
        end_value = _timestamp_to_input_value(max_ts)
        day_options = _build_day_options(min_ts, max_ts)
        day_default = day_options[0]["value"] if day_options else None

        payload = {
            "db_path": db_path_to_use,
            "filename": display_name or Path(db_path_to_use).name,
            "min_ts": min_ts,
            "max_ts": max_ts,
            "channel_labels": {column: label for column, label in available_channels},
        }

        return (
            payload,
            f"Файл: {payload['filename']}",
            _build_meta_line(min_ts=min_ts, max_ts=max_ts, channel_count=len(available_channels)),
            options,
            default_selection,
            start_value,
            end_value,
            start_value,
            end_value,
            start_value,
            end_value,
            day_options,
            day_default,
        )
    except Exception as error:
        _cleanup_temp_db(temp_db_path)
        if isinstance(db_store, dict):
            return (
                no_update,
                f"Ошибка загрузки: {error}",
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
            )
        return (
            None,
            f"Ошибка загрузки: {error}",
            "Диапазон: — | Каналов: —",
            [],
            [],
            None,
            None,
            None,
            None,
            None,
            None,
            [],
            None,
        )


@callback(
    Output("channels-checklist", "value"),
    Input("select-all-btn", "n_clicks"),
    Input("clear-btn", "n_clicks"),
    Input("build-all-btn", "n_clicks"),
    State("channels-checklist", "options"),
    prevent_initial_call=True,
)
def update_channel_selection(
    _select_all_clicks: Optional[int],
    _clear_clicks: Optional[int],
    _build_all_clicks: Optional[int],
    channel_options: Optional[list[dict[str, str]]],
):
    available_options = channel_options or []
    if not available_options:
        return []

    trigger = _current_triggered_id()
    if trigger in {"select-all-btn", "build-all-btn"}:
        return [option["value"] for option in available_options]
    if trigger == "clear-btn":
        return []
    raise PreventUpdate


@callback(
    Output("start-dt", "value"),
    Output("end-dt", "value"),
    Input("apply-day-btn", "n_clicks"),
    Input("full-range-btn", "n_clicks"),
    State("day-dropdown", "value"),
    State("db-store", "data"),
    prevent_initial_call=True,
)
def apply_time_shortcuts(
    _apply_day_clicks: Optional[int],
    _full_range_clicks: Optional[int],
    selected_day: Optional[str],
    db_store: Optional[dict[str, Any]],
):
    if not isinstance(db_store, dict):
        raise PreventUpdate

    min_ts = float(db_store["min_ts"])
    max_ts = float(db_store["max_ts"])
    trigger = _current_triggered_id()

    if trigger == "full-range-btn":
        return _timestamp_to_input_value(min_ts), _timestamp_to_input_value(max_ts)

    if trigger == "apply-day-btn" and selected_day:
        try:
            day_start = datetime.strptime(selected_day, "%Y-%m-%d")
        except ValueError:
            raise PreventUpdate
        day_end = day_start + timedelta(days=1) - timedelta(milliseconds=1)
        start_ts = max(min_ts, day_start.timestamp())
        end_ts = min(max_ts, day_end.timestamp())
        return _timestamp_to_input_value(start_ts), _timestamp_to_input_value(end_ts)

    raise PreventUpdate


@callback(
    Output("main-chart", "figure"),
    Output("traces-store", "data"),
    Output("status-line", "children"),
    Input("db-store", "data"),
    Input("build-selected-btn", "n_clicks"),
    Input("build-all-btn", "n_clicks"),
    Input("show-y-axes", "value"),
    Input("channels-checklist", "value"),
    Input("start-dt", "value"),
    Input("end-dt", "value"),
    Input("limit-input", "value"),
    State("channels-checklist", "options"),
)
def build_chart(
    db_store: Optional[dict[str, Any]],
    _selected_clicks: Optional[int],
    _all_clicks: Optional[int],
    show_y_axes_values: list[str],
    selected_values: Optional[list[str]],
    start_value: Optional[str],
    end_value: Optional[str],
    limit_value: Optional[float],
    channel_options: Optional[list[dict[str, str]]],
):
    if not isinstance(db_store, dict):
        return (
            _build_placeholder_figure("Откройте файл базы данных и постройте график."),
            [],
            "Ожидание загрузки базы данных.",
        )

    db_path = db_store.get("db_path")
    if not db_path or not Path(db_path).exists():
        return (
            _build_placeholder_figure("Файл базы данных не найден. Загрузите его повторно."),
            [],
            "Временный файл базы данных недоступен.",
        )

    trigger = _current_triggered_id()
    available_options = channel_options or []
    selected_columns = list(selected_values or [])
    if trigger == "build-all-btn":
        selected_columns = [option["value"] for option in available_options]

    if not selected_columns:
        return (
            _build_placeholder_figure("Выберите хотя бы один канал Y."),
            [],
            "Каналы не выбраны.",
        )

    start_ts = _input_value_to_timestamp(start_value)
    end_ts = _input_value_to_timestamp(end_value)
    min_ts = float(db_store["min_ts"])
    max_ts = float(db_store["max_ts"])
    if start_ts is None:
        start_ts = min_ts
    if end_ts is None:
        end_ts = max_ts
    if start_ts > end_ts:
        return (
            _build_placeholder_figure("Начало должно быть меньше или равно концу."),
            [],
            "Ошибка: время начала больше времени конца.",
        )
    start_ts = max(start_ts, min_ts)
    end_ts = min(end_ts, max_ts)

    try:
        limit = int(float(limit_value)) if limit_value is not None else 4511
    except (TypeError, ValueError):
        limit = 4511
    limit = max(10, min(limit, 1_000_000))

    label_map: dict[str, str] = dict(db_store.get("channel_labels") or _db_channel_labels())
    selected_channels = [(column, label_map.get(column, column)) for column in selected_columns]

    try:
        service = SQLiteService(str(db_path))
        traces_payload, points_total = _build_traces_payload(
            service=service,
            selected_channels=selected_channels,
            limit=limit,
            start_ts=start_ts,
            end_ts=end_ts,
        )
    except sqlite3.Error as error:
        return (
            _build_placeholder_figure("Ошибка чтения из базы данных."),
            [],
            f"Ошибка чтения данных: {error}",
        )
    except Exception as error:
        return (
            _build_placeholder_figure("Ошибка подготовки графика."),
            [],
            f"Ошибка: {error}",
        )

    if not traces_payload:
        return (
            _build_placeholder_figure("Нет данных в выбранном диапазоне."),
            [],
            "В выбранном диапазоне нет точек для отображения.",
        )

    show_y_axes = "show" in (show_y_axes_values or [])
    figure = _build_plotly_figure(
        traces_payload=traces_payload,
        start_ts=start_ts,
        end_ts=end_ts,
        show_y_axes=show_y_axes,
    )
    status = f"Построено каналов: {len(traces_payload)} | Точек: {points_total} | Лимит: {limit}"
    return figure, traces_payload, status


@callback(
    Output("clicked-time", "children"),
    Output("clicked-values", "children"),
    Input("main-chart", "clickData"),
    Input("traces-store", "data"),
)
def update_clicked_values(
    click_data: Optional[dict[str, Any]],
    traces_payload: Optional[list[dict[str, Any]]],
):
    trigger = _current_triggered_id()
    if trigger == "traces-store" or not click_data or not traces_payload:
        return "Время X: —", []

    points = click_data.get("points") or []
    if not points:
        return "Время X: —", []

    x_raw = points[0].get("x")
    x_ts = _parse_plotly_x_to_timestamp(str(x_raw))
    if x_ts is None:
        return f"Время X: не распознано ({x_raw})", []

    clicked_time = datetime.fromtimestamp(x_ts).strftime("%d.%m.%Y %H:%M:%S.%f")[:-3]
    lines: list[html.Li] = []

    for payload in traces_payload:
        ts_values = payload.get("ts_values") or []
        y_values = payload.get("y_values") or []
        if not ts_values or not y_values:
            continue

        idx = bisect_left(ts_values, x_ts)
        candidates = []
        if idx < len(ts_values):
            candidates.append(idx)
        if idx > 0:
            candidates.append(idx - 1)
        if not candidates:
            continue

        best_idx = min(candidates, key=lambda i: abs(ts_values[i] - x_ts))
        y_value = y_values[best_idx]
        name = _strip_type_suffix(str(payload.get("name", "")))
        lines.append(html.Li(f"{name}: {_format_number(float(y_value))}"))

    return f"Время X: {clicked_time}", lines


def main() -> None:
    host = os.environ.get("DASH_HOST", "127.0.0.1")
    port = int(os.environ.get("DASH_PORT", "8050"))
    open_browser = os.environ.get("DASH_OPEN_BROWSER", "0") == "1"
    if open_browser:
        timer = Timer(1.0, _try_open_browser, args=(host, port))
        timer.daemon = True
        timer.start()
    app.run(host=host, port=port, debug=False)


def _try_open_browser(host: str, port: int) -> None:
    try:
        webbrowser.open(f"http://{host}:{port}")
    except Exception:
        return


if __name__ == "__main__":
    main()
