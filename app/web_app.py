import base64
import math
import os
import sqlite3
import sys
import tempfile
import webbrowser
from bisect import bisect_left
from datetime import datetime, timedelta
from pathlib import Path
from threading import Timer
from typing import Any, Optional

import plotly.graph_objects as go
try:
    from dash import Dash, Input, Output, State, ctx, dcc, html, no_update
    from dash.exceptions import PreventUpdate
except ImportError as error:  # pragma: no cover - runtime dependency guard
    raise SystemExit(
        "Не найден пакет 'dash'. Установите зависимости: pip install -r requirements.txt"
    ) from error

from app.data.sqlite_service import SQLiteService, quote_identifier


DATA_TABLE = "data"
TIME_COLUMN = "time@timestamp"
MAX_RENDER_POINTS_PER_CHANNEL = 3500
MAX_TOTAL_RENDER_POINTS = 220_000
MAX_UPLOAD_FILES = 32
INITIAL_X_WINDOW_SECONDS = 6 * 60 * 60
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


def _extract_click_timestamp(
    click_data: dict[str, Any],
    traces_payload: list[dict[str, Any]],
) -> Optional[float]:
    points = click_data.get("points") or []
    if not points:
        return None

    point = points[0] or {}
    x_raw = point.get("x")
    if x_raw is not None:
        parsed = _parse_plotly_x_to_timestamp(str(x_raw))
        if parsed is not None:
            return parsed

    curve_number = point.get("curveNumber")
    point_number = point.get("pointNumber", point.get("pointIndex"))
    if not isinstance(curve_number, int) or not isinstance(point_number, int):
        return None
    if curve_number < 0 or curve_number >= len(traces_payload):
        return None

    ts_values = traces_payload[curve_number].get("ts_values") or []
    if point_number < 0 or point_number >= len(ts_values):
        return None
    return _to_float_or_none(ts_values[point_number])


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


def _build_meta_line(min_ts: float, max_ts: float, channel_count: int, file_count: int) -> str:
    start_text = datetime.fromtimestamp(min_ts).strftime("%d.%m.%Y %H:%M:%S")
    end_text = datetime.fromtimestamp(max_ts).strftime("%d.%m.%Y %H:%M:%S")
    return f"Файлов: {file_count} | Диапазон: {start_text} - {end_text} | Каналов: {channel_count}"


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _extract_temp_paths_from_store(db_store: Optional[dict[str, Any]]) -> list[str]:
    if not isinstance(db_store, dict):
        return []

    paths: list[str] = []
    db_path = db_store.get("db_path")
    if isinstance(db_path, str) and db_path:
        paths.append(db_path)

    raw_sources = db_store.get("sources")
    if isinstance(raw_sources, list):
        for source in raw_sources:
            if not isinstance(source, dict):
                continue
            source_path = source.get("db_path")
            if isinstance(source_path, str) and source_path:
                paths.append(source_path)

    return paths


def _build_available_channels(sources: list[dict[str, Any]]) -> list[tuple[str, str]]:
    if not sources:
        return []

    common_columns: Optional[set[str]] = None
    for source in sources:
        labels_map = source.get("channel_labels") or {}
        source_columns = {str(column) for column in labels_map.keys()}
        common_columns = source_columns if common_columns is None else common_columns & source_columns

    if not common_columns:
        return []
    return [(column, label) for column, label in Y_CHANNELS if column in common_columns]


def _build_db_info_line(
    loaded_sources: list[dict[str, Any]],
    skipped_count: int,
    was_limited: bool,
) -> str:
    if not loaded_sources:
        return "Файлы не выбраны"

    names = [str(source.get("filename", "без имени")) for source in loaded_sources]
    preview = ", ".join(names[:3])
    if len(names) > 3:
        preview += ", ..."
    parts = [f"Файлов: {len(loaded_sources)}", preview]
    if was_limited:
        parts.append(f"взяты первые {MAX_UPLOAD_FILES}")
    if skipped_count > 0:
        parts.append(f"пропущено: {skipped_count}")
    return " | ".join(parts)


def _inspect_uploaded_source(db_path: str, display_name: str) -> dict[str, Any]:
    service = SQLiteService(db_path)
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

    return {
        "db_path": db_path,
        "filename": display_name or Path(db_path).name,
        "min_ts": min_ts,
        "max_ts": max_ts,
        "channel_labels": {column: label for column, label in available_channels},
    }


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


def _merge_sources_into_temp_db(
    sources: list[dict[str, Any]],
    common_columns: list[str],
) -> tuple[str, int]:
    with tempfile.NamedTemporaryFile(
        mode="wb",
        suffix=".db",
        prefix="dash_chart_merged_",
        delete=False,
    ) as temp_file:
        merged_path = temp_file.name

    q_table = quote_identifier(DATA_TABLE)
    q_time = quote_identifier(TIME_COLUMN)
    quoted_columns = [quote_identifier(column) for column in common_columns]
    create_columns = [f"{q_time} REAL"] + [f"{quoted} REAL" for quoted in quoted_columns]
    create_sql = f"CREATE TABLE {q_table} ({', '.join(create_columns)})"

    insert_columns = [q_time] + quoted_columns
    placeholders = ", ".join(["?"] * len(insert_columns))
    insert_sql = f"INSERT INTO {q_table} ({', '.join(insert_columns)}) VALUES ({placeholders})"

    rows_written = 0
    connection = sqlite3.connect(merged_path)
    try:
        connection.execute(create_sql)
        for source in sources:
            source_path = str(source["db_path"])
            source_connection = sqlite3.connect(source_path)
            try:
                select_sql = (
                    f"SELECT {q_time}, {', '.join(quoted_columns)} "
                    f"FROM {q_table} "
                    f"WHERE {q_time} IS NOT NULL "
                    f"ORDER BY {q_time}"
                )
                cursor = source_connection.execute(select_sql)
                while True:
                    batch = cursor.fetchmany(5000)
                    if not batch:
                        break
                    connection.executemany(insert_sql, batch)
                    rows_written += len(batch)
            finally:
                source_connection.close()

        index_sql = f"CREATE INDEX IF NOT EXISTS idx_data_time_timestamp_ts ON {q_table} ({q_time})"
        connection.execute(index_sql)
        connection.commit()
    except Exception:
        connection.close()
        _cleanup_temp_db(merged_path)
        raise

    connection.close()
    return merged_path, rows_written


def _compute_downsample_target(limit: int, source_count: int, channel_count: int) -> int:
    trace_count = max(1, source_count * channel_count)
    total_budget_per_trace = max(250, MAX_TOTAL_RENDER_POINTS // trace_count)
    return max(250, min(limit, MAX_RENDER_POINTS_PER_CHANNEL, total_budget_per_trace))


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
    downsample_target: int,
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
            "channel_name": _strip_type_suffix(label),
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

    traces_payload: list[dict[str, object]] = []
    points_total = 0

    for column_name, _ in selected_channels:
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
                "name": buffer["channel_name"],
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
    selected_columns: list[str],
    channel_label_map: dict[str, str],
    start_ts: float,
    end_ts: float,
    show_y_axes: bool,
    file_count: int,
) -> go.Figure:
    figure = go.Figure()

    active_columns = [column for column in selected_columns if any(p.get("column") == column for p in traces_payload)]
    if not active_columns:
        active_columns = []
        for payload in traces_payload:
            column = str(payload.get("column", ""))
            if column and column not in active_columns:
                active_columns.append(column)

    axis_index_by_column = {column: index + 1 for index, column in enumerate(active_columns)}

    for payload in traces_payload:
        column = str(payload.get("column", ""))
        axis_index = axis_index_by_column.get(column, 1)
        yaxis_name = "y" if axis_index == 1 else f"y{axis_index}"
        x_values = [datetime.fromtimestamp(ts) for ts in payload["ts_values"]]  # type: ignore[index]
        figure.add_trace(
            go.Scatter(
                x=x_values,
                y=payload["y_values"],  # type: ignore[index]
                mode="lines+markers",
                name=payload["name"],  # type: ignore[index]
                line={
                    "color": payload["color"],  # type: ignore[index]
                    "width": 1.8,
                    "simplify": False,
                },
                marker={"size": 9, "opacity": 0.001},
                yaxis=yaxis_name,
                hovertemplate="%{x|%d.%m.%Y %H:%M:%S}<br>%{y:.6g}<extra></extra>",
            )
        )

    if show_y_axes:
        axis_count = len(active_columns)
        left_axes = [idx for idx in range(1, axis_count + 1) if idx % 2 == 1]
        right_axes = [idx for idx in range(1, axis_count + 1) if idx % 2 == 0]
        left_extra = left_axes[1:]
        right_extra = right_axes

        left_pad = min(0.045 + 0.016 * len(left_extra), 0.12)
        right_pad = min(0.038 + 0.016 * len(right_extra), 0.12)
        if left_pad + right_pad > 0.24:
            scale = 0.24 / (left_pad + right_pad)
            left_pad *= scale
            right_pad *= scale
        x_domain = [left_pad, 1.0 - right_pad]

        left_positions: dict[int, float] = {1: x_domain[0]}
        right_positions: dict[int, float] = {}
        for order, idx in enumerate(left_extra, start=1):
            left_positions[idx] = max(0.0, x_domain[0] - 0.016 * order)
        for order, idx in enumerate(right_extra, start=1):
            right_positions[idx] = min(1.0, x_domain[1] + 0.016 * order)
    else:
        x_domain = [0.028, 0.997]
        left_positions = {}
        right_positions = {}

    layout_axes = {}
    for axis_index, column in enumerate(active_columns, start=1):
        axis_key = "yaxis" if axis_index == 1 else f"yaxis{axis_index}"
        axis_color = SERIES_COLORS[(axis_index - 1) % len(SERIES_COLORS)]

        fixed_range = FIXED_Y_RANGES.get(column)
        if fixed_range is not None:
            y_min, y_max = fixed_range
        else:
            mins = [float(trace["y_min"]) for trace in traces_payload if trace.get("column") == column]
            maxs = [float(trace["y_max"]) for trace in traces_payload if trace.get("column") == column]
            y_min, y_max = _normalized_numeric_range(min(mins), max(maxs))

        base_cfg = {
            "range": [y_min, y_max],
            "zeroline": False,
            "showline": True,
            "linecolor": "#c8d4e2",
            "linewidth": 1,
        }
        if show_y_axes:
            channel_title = _strip_type_suffix(channel_label_map.get(column, column))
            base_cfg.update(
                {
                    "title": {"text": channel_title, "font": {"color": axis_color, "size": 9}, "standoff": 2},
                    "tickfont": {"color": axis_color, "size": 9},
                    "ticks": "outside",
                    "ticklen": 2,
                    "tickwidth": 1,
                    "nticks": 5,
                }
            )
        else:
            base_cfg.update({"visible": False, "showgrid": False, "showticklabels": False})

        if axis_index == 1:
            base_cfg.update({"showgrid": show_y_axes})
        else:
            side = "left" if axis_index in left_positions else "right"
            position = left_positions.get(axis_index, right_positions.get(axis_index, x_domain[1]))
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
    initial_end_ts = min(end_ts, start_ts + INITIAL_X_WINDOW_SECONDS)
    initial_end_dt = datetime.fromtimestamp(initial_end_ts)
    figure.update_layout(
        title=(
            f"Файлов: {file_count} | Каналов: {len(active_columns)} | "
            f"{start_dt.strftime('%d.%m.%Y %H:%M:%S')} - "
            f"{end_dt.strftime('%d.%m.%Y %H:%M:%S')}"
        ),
        xaxis={
            "title": {"text": "Время"},
            "type": "date",
            "domain": x_domain,
            "range": [start_dt, initial_end_dt],
            "showgrid": True,
            "rangeslider": {"visible": False},
            "showspikes": False,
        },
        hovermode="closest",
        clickmode="event+select",
        hoverdistance=-1,
        spikedistance=-1,
        legend={"orientation": "h", "y": -0.2, "x": 0.0, "font": {"size": 9}},
        margin={"l": 28, "r": 28, "t": 70, "b": 105},
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
                    className="panel-card",
                    children=[
                        dcc.Upload(
                            id="db-upload",
                            className="upload-area",
                            children=html.Div(
                                [
                                    html.Span("Перетащите до 32 .db файлов сюда или "),
                                    html.Span("выберите файлы", className="linkish"),
                                ]
                            ),
                            multiple=True,
                            accept=".db,.sqlite,.sqlite3,application/octet-stream",
                        ),
                        html.Div("Файлы не выбраны", id="db-info", className="db-info"),
                        html.Div("Файлов: — | Диапазон: — | Каналов: —", id="meta-line", className="meta-line"),
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
                html.Label("Строк на канал (на файл)", className="input-label"),
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
                html.Button("Сохранить график (PNG)", id="save-chart-btn", className="ghost-btn wide-btn"),
                html.Div("Сохранение выполняется в загрузки браузера.", id="save-msg", className="hint"),
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
                            figure=_build_placeholder_figure("Загрузите до 32 файлов базы данных и постройте график."),
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


@app.callback(
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
    State("db-upload", "filename"),
    State("db-store", "data"),
    prevent_initial_call=True,
)
def load_database(
    contents: Optional[Any],
    filename: Optional[Any],
    db_store: Optional[dict[str, Any]],
):
    upload_contents = [str(item) for item in _as_list(contents) if isinstance(item, str) and item]
    if not upload_contents:
        raise PreventUpdate

    upload_names = _as_list(filename)
    was_limited = len(upload_contents) > MAX_UPLOAD_FILES
    upload_contents = upload_contents[:MAX_UPLOAD_FILES]

    previous_paths = _extract_temp_paths_from_store(db_store)
    created_paths: list[str] = []
    loaded_sources: list[dict[str, Any]] = []
    merged_db_path: Optional[str] = None
    skipped_count = 0

    try:
        for index, upload_content in enumerate(upload_contents):
            temp_db_path: Optional[str] = None
            try:
                temp_db_path = _decode_upload_to_temp_db(upload_content)
                created_paths.append(temp_db_path)

                raw_name = upload_names[index] if index < len(upload_names) else None
                display_name = str(raw_name).strip() if isinstance(raw_name, str) and raw_name.strip() else ""
                if not display_name:
                    display_name = Path(temp_db_path).name

                source = _inspect_uploaded_source(temp_db_path, display_name)
                loaded_sources.append(source)
            except Exception:
                skipped_count += 1
                _cleanup_temp_db(temp_db_path)
                if temp_db_path and temp_db_path in created_paths:
                    created_paths.remove(temp_db_path)

        if not loaded_sources:
            raise ValueError("Не удалось загрузить ни одного корректного файла .db.")

        available_channels = _build_available_channels(loaded_sources)
        if not available_channels:
            raise ValueError("В загруженных файлах нет общих поддерживаемых каналов Y.")

        common_columns = [column for column, _ in available_channels]
        merged_db_path, merged_rows = _merge_sources_into_temp_db(loaded_sources, common_columns)

        merged_service = SQLiteService(merged_db_path)
        bounds = merged_service.fetch_time_bounds(DATA_TABLE, TIME_COLUMN)
        if not bounds:
            raise ValueError("После объединения не найдено данных для построения графика.")
        min_ts, max_ts = bounds

        for source_path in created_paths:
            _cleanup_temp_db(source_path)
        created_paths = []

        for old_path in previous_paths:
            if old_path != merged_db_path:
                _cleanup_temp_db(old_path)

        options = [{"label": label, "value": column} for column, label in available_channels]
        default_selection = [available_channels[0][0]]
        start_value = _timestamp_to_input_value(min_ts)
        end_value = _timestamp_to_input_value(max_ts)
        day_options = _build_day_options(min_ts, max_ts)
        day_default = day_options[0]["value"] if day_options else None

        payload = {
            "db_path": merged_db_path,
            "filename": f"Объединенная база ({len(loaded_sources)} файлов)",
            "file_count": len(loaded_sources),
            "merged_rows": merged_rows,
            "source_names": [str(source.get("filename", "")) for source in loaded_sources],
            "min_ts": min_ts,
            "max_ts": max_ts,
            "channel_labels": {column: label for column, label in available_channels},
        }

        return (
            payload,
            (
                _build_db_info_line(loaded_sources, skipped_count=skipped_count, was_limited=was_limited)
                + f" | объединено строк: {merged_rows}"
            ),
            _build_meta_line(
                min_ts=min_ts,
                max_ts=max_ts,
                channel_count=len(available_channels),
                file_count=len(loaded_sources),
            ),
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
        for path in created_paths:
            _cleanup_temp_db(path)
        _cleanup_temp_db(merged_db_path)
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
            "Файлов: — | Диапазон: — | Каналов: —",
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


@app.callback(
    Output("channels-checklist", "value", allow_duplicate=True),
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


@app.callback(
    Output("start-dt", "value", allow_duplicate=True),
    Output("end-dt", "value", allow_duplicate=True),
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


@app.callback(
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
            _build_placeholder_figure("Загрузите до 32 файлов базы данных и постройте график."),
            [],
            "Ожидание загрузки базы данных.",
        )

    db_path = db_store.get("db_path")
    if not isinstance(db_path, str) or not db_path:
        return (
            _build_placeholder_figure("Базы данных не загружены."),
            [],
            "Нет доступных файлов базы данных.",
        )

    if not Path(db_path).exists():
        return (
            _build_placeholder_figure("Объединенная база недоступна. Загрузите файлы повторно."),
            [],
            "Временный файл объединенной базы недоступен.",
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
    downsample_target = _compute_downsample_target(
        limit=limit,
        source_count=1,
        channel_count=len(selected_channels),
    )

    try:
        service = SQLiteService(db_path)
        traces_payload, points_total = _build_traces_payload(
            service=service,
            selected_channels=selected_channels,
            limit=limit,
            start_ts=start_ts,
            end_ts=end_ts,
            downsample_target=downsample_target,
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
        selected_columns=selected_columns,
        channel_label_map=label_map,
        start_ts=start_ts,
        end_ts=end_ts,
        show_y_axes=show_y_axes,
        file_count=int(db_store.get("file_count") or 1),
    )
    status_parts = [
        f"Файлов: {int(db_store.get('file_count') or 1)}",
        f"Каналов: {len(selected_channels)}",
        f"Трейсов: {len(traces_payload)}",
        f"Точек: {points_total}",
        f"Лимит строк: {limit}",
        f"Точек/трейс: {downsample_target}",
    ]
    status = " | ".join(status_parts)
    return figure, traces_payload, status


@app.callback(
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

    x_ts = _extract_click_timestamp(click_data, traces_payload)
    if x_ts is None:
        return "Время X: не распознано", []

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


app.clientside_callback(
    """
    function(n_clicks, figure) {
        if (!n_clicks) {
            return window.dash_clientside.no_update;
        }
        if (!figure || !figure.data || figure.data.length === 0) {
            return "Нет графика для сохранения.";
        }

        const plotEl = document.querySelector("#main-chart .js-plotly-plot");
        if (!plotEl || !window.Plotly) {
            return "График еще не готов. Постройте график и повторите.";
        }

        const stamp = new Date().toISOString().replace(/[:.]/g, "-");
        const fileName = "grafik-" + stamp;
        window.Plotly.downloadImage(plotEl, {
            format: "png",
            filename: fileName,
            scale: 2
        });
        return "Сохранено: " + fileName + ".png";
    }
    """,
    Output("save-msg", "children"),
    Input("save-chart-btn", "n_clicks"),
    State("main-chart", "figure"),
    prevent_initial_call=True,
)


def main() -> None:
    host = os.environ.get("DASH_HOST", "127.0.0.1")
    port = int(os.environ.get("DASH_PORT", "8050"))
    default_open = "1" if getattr(sys, "frozen", False) else "0"
    open_browser = os.environ.get("DASH_OPEN_BROWSER", default_open) == "1"
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
