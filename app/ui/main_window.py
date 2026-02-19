import json
import math
import sqlite3
import tempfile
from bisect import bisect_left
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from PySide6.QtCore import (
    QDateTime,
    QObject,
    QSignalBlocker,
    QThread,
    QTimer,
    QUrl,
    Qt,
    Signal,
    Slot,
)
from PySide6.QtGui import QAction
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineCore import QWebEngineSettings
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDateEdit,
    QDateTimeEdit,
    QFrame,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

try:
    import plotly
    import plotly.graph_objects as go
    from plotly.utils import PlotlyJSONEncoder
except ImportError:  # pragma: no cover - depends on environment
    plotly = None
    go = None
    PlotlyJSONEncoder = None

from app.data.sqlite_service import SQLiteService


DATA_TABLE = "data"
TIME_COLUMN = "time@timestamp"
PLOTLY_AVAILABLE = go is not None and PlotlyJSONEncoder is not None
MAX_RENDER_POINTS_PER_CHANNEL = 3500
CACHE_MAX_ITEMS = 16
HEALTH_CHECK_MAX_ATTEMPTS = 40
HEALTH_CHECK_RETRY_MS = 250

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
    "#E63946",
    "#1D3557",
    "#2A9D8F",
    "#F4A261",
    "#E9C46A",
    "#118AB2",
    "#8338EC",
    "#EF476F",
]

FIXED_Y_RANGES: dict[str, tuple[float, float]] = {
    "data_format_0": (0.0, 50.0),   # Объект мойки
    "data_format_1": (0.0, 20.0),   # Программа мойки
    "data_format_2": (0.0, 50.0),   # Процесс
    "data_format_3": (0.0, 50.0),   # Расход подачи
    "data_format_4": (0.0, 20.0),   # Давление подачи
    "data_format_5": (0.0, 100.0),  # Температура подачи
    "data_format_6": (0.0, 100.0),  # Температура возврата
    "data_format_7": (0.0, 5.0),    # Концентрация возврата
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


class ChartBuildThread(QThread):
    build_succeeded = Signal(int, object, int)
    build_failed = Signal(int, str)

    def __init__(
        self,
        request_id: int,
        db_path: str,
        selected_channels: list[tuple[str, str]],
        limit: int,
        start_ts: float,
        end_ts: float,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._request_id = request_id
        self._db_path = db_path
        self._selected_channels = selected_channels
        self._limit = int(limit)
        self._start_ts = float(start_ts)
        self._end_ts = float(end_ts)

    def run(self) -> None:
        try:
            service = SQLiteService(self._db_path)
            channel_columns = [column for column, _ in self._selected_channels]
            rows = service.fetch_multi_series_rows(
                table_name=DATA_TABLE,
                y_columns=channel_columns,
                limit=self._limit,
                start_ts=self._start_ts,
                end_ts=self._end_ts,
            )

            buffers: dict[str, dict[str, object]] = {}
            for index, (column_name, label) in enumerate(self._selected_channels):
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

                for column_name, _ in self._selected_channels:
                    y_value = _to_float_or_none(row[column_name])
                    if y_value is None:
                        continue

                    channel_buffer = buffers[column_name]
                    channel_buffer["ts_values"].append(ts_seconds)  # type: ignore[index]
                    channel_buffer["y_values"].append(y_value)      # type: ignore[index]
                    channel_buffer["min_y"] = min(channel_buffer["min_y"], y_value)  # type: ignore[index]
                    channel_buffer["max_y"] = max(channel_buffer["max_y"], y_value)  # type: ignore[index]

            downsample_target = max(500, min(self._limit, MAX_RENDER_POINTS_PER_CHANNEL))
            traces_payload: list[dict] = []
            points_total = 0

            for column_name, label in self._selected_channels:
                channel_buffer = buffers[column_name]
                ts_values: list[float] = channel_buffer["ts_values"]  # type: ignore[assignment]
                y_values: list[float] = channel_buffer["y_values"]    # type: ignore[assignment]
                if not ts_values:
                    continue

                sampled_ts, sampled_y = _downsample_lttb(ts_values, y_values, downsample_target)
                x_values = [datetime.fromtimestamp(ts) for ts in sampled_ts]

                fixed_range = FIXED_Y_RANGES.get(column_name)
                if fixed_range is None:
                    min_y = float(channel_buffer["min_y"])  # type: ignore[arg-type]
                    max_y = float(channel_buffer["max_y"])  # type: ignore[arg-type]
                    y_min, y_max = _normalized_numeric_range(min_y, max_y)
                else:
                    y_min, y_max = fixed_range

                traces_payload.append(
                    {
                        "column": column_name,
                        "name": label,
                        "color": channel_buffer["color"],
                        "x_values": x_values,
                        "ts_values": sampled_ts,
                        "y_values": sampled_y,
                        "y_min": y_min,
                        "y_max": y_max,
                    }
                )
                points_total += len(sampled_ts)

            self.build_succeeded.emit(self._request_id, traces_payload, points_total)
        except sqlite3.Error as error:
            self.build_failed.emit(self._request_id, f"Ошибка чтения данных:\n{error}")
        except Exception as error:  # pragma: no cover - worker runtime path
            self.build_failed.emit(self._request_id, f"Ошибка подготовки графика:\n{error}")


class PlotlyBridge(QObject):
    point_clicked = Signal(str)

    @Slot(str)
    def onPlotClicked(self, x_value: str) -> None:
        self.point_clicked.emit(x_value)


class PlotlyChartView(QWidget):
    point_clicked = Signal(str)
    render_error = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._web_view = QWebEngineView(self)
        settings = self._web_view.settings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)
        settings.setAttribute(
            QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls,
            True,
        )
        settings.setAttribute(
            QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls,
            True,
        )
        layout.addWidget(self._web_view, 1)

        self._bridge = PlotlyBridge(self)
        self._bridge.point_clicked.connect(self.point_clicked.emit)
        self._channel = QWebChannel(self._web_view.page())
        self._channel.registerObject("bridge", self._bridge)
        self._web_view.page().setWebChannel(self._channel)
        self._web_view.loadFinished.connect(self._on_html_loaded)

        self._has_data = False
        self._html_file_path: Optional[Path] = None
        self._last_figure = None
        self._is_health_check_pending = False
        self._health_check_attempt = 0
        self._fallback_to_embedded_js = False
        self._react_request_id = 0
        self.set_placeholder("Откройте базу данных и постройте график.")

    def has_data(self) -> bool:
        return self._has_data

    def set_placeholder(self, title: str) -> None:
        self._has_data = False
        self._last_figure = None
        self._is_health_check_pending = False
        self._health_check_attempt = 0
        self._fallback_to_embedded_js = False
        self._react_request_id = 0
        html = f"""
        <html>
          <head>
            <meta charset="utf-8" />
            <style>
              body {{
                margin: 0;
                font-family: Arial, sans-serif;
                background: #ffffff;
                color: #374151;
                display: flex;
                align-items: center;
                justify-content: center;
                height: 100vh;
              }}
              .title {{
                font-size: 18px;
                text-align: center;
                padding: 24px;
              }}
            </style>
          </head>
          <body>
            <div class="title">{title}</div>
          </body>
        </html>
        """
        self._web_view.setHtml(html)

    def set_figure(self, figure) -> None:
        self._last_figure = figure
        if self._has_data and not self._is_health_check_pending:
            self._react_figure(figure)
            return
        self._fallback_to_embedded_js = False
        self._load_figure(embed_js=False)

    def reset_view(self) -> None:
        if not self._has_data:
            return
        script = """
        (function() {
          if (typeof Plotly === 'undefined') return;
          var plots = document.querySelectorAll('.plotly-graph-div');
          if (!plots.length) return;
          var plot = plots[0];
          var update = {'xaxis.autorange': true};
          var yAxes = Object.keys(plot.layout).filter(function(k){ return /^yaxis\\d*$/.test(k); });
          yAxes.forEach(function(axisKey) {
            update[axisKey + '.autorange'] = true;
          });
          Plotly.relayout(plot, update);
        })();
        """
        self._web_view.page().runJavaScript(script)

    def grab_chart(self):
        return self._web_view.grab()

    def _load_figure(self, embed_js: bool) -> None:
        if self._last_figure is None:
            return

        include_plotlyjs: str | bool = True
        if not embed_js:
            local_js_url = self._resolve_local_plotly_js_url()
            if local_js_url:
                include_plotlyjs = local_js_url

        html = self._last_figure.to_html(
            include_plotlyjs=include_plotlyjs,
            full_html=True,
            config=self._plotly_config(),
        )
        html = self._inject_bridge_script(html)
        self._write_html_to_temp_file(html)
        self._is_health_check_pending = True
        self._health_check_attempt = 0
        self._has_data = False
        self._web_view.load(QUrl.fromLocalFile(str(self._html_file_path)))

    def _react_figure(self, figure) -> None:
        if PlotlyJSONEncoder is None:
            self._fallback_to_embedded_js = False
            self._load_figure(embed_js=False)
            return

        payload = json.dumps(figure.to_plotly_json(), cls=PlotlyJSONEncoder)
        config_json = json.dumps(self._plotly_config())
        self._react_request_id += 1
        request_id = self._react_request_id
        script = f"""
        (function() {{
          if (typeof Plotly === 'undefined') return false;
          var plot = document.querySelector('.plotly-graph-div');
          if (!plot) return false;
          var fig = {payload};
          var cfg = {config_json};
          Plotly.react(plot, fig.data || [], fig.layout || {{}}, cfg);
          return true;
        }})();
        """

        def _on_react_done(ok: Any) -> None:
            if request_id != self._react_request_id:
                return
            if ok:
                self._has_data = True
                return
            self._fallback_to_embedded_js = False
            self._load_figure(embed_js=False)

        self._web_view.page().runJavaScript(script, _on_react_done)

    def _write_html_to_temp_file(self, html: str) -> None:
        if self._html_file_path and self._html_file_path.exists():
            try:
                self._html_file_path.unlink()
            except OSError:
                pass

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".html",
            prefix="plotly_chart_",
            delete=False,
        ) as temp_file:
            temp_file.write(html)
            self._html_file_path = Path(temp_file.name)

    def _on_html_loaded(self, ok: bool) -> None:
        if not self._is_health_check_pending:
            return

        if not ok:
            self._is_health_check_pending = False
            self._has_data = False
            self.render_error.emit("Не удалось загрузить HTML-график в Qt WebEngine.")
            return

        QTimer.singleShot(120, self._run_health_check)

    def _run_health_check(self) -> None:
        if not self._is_health_check_pending:
            return

        script = """
        (function() {
          var plot = document.querySelector('.plotly-graph-div');
          var traceCount = 0;
          if (plot && Array.isArray(plot.data)) {
            traceCount = plot.data.length;
          }
          return {
            hasPlotly: typeof window.Plotly !== 'undefined',
            hasPlot: !!plot,
            traceCount: traceCount,
            lastError: window.__plotlyLastError || null
          };
        })();
        """
        self._web_view.page().runJavaScript(script, self._on_health_check_result)

    def _on_health_check_result(self, payload: Any) -> None:
        has_plotly = False
        has_plot = False
        trace_count = 0
        last_error = ""

        if isinstance(payload, dict):
            has_plotly = bool(payload.get("hasPlotly"))
            has_plot = bool(payload.get("hasPlot"))
            try:
                trace_count = int(payload.get("traceCount", 0))
            except (TypeError, ValueError):
                trace_count = 0
            last_error = str(payload.get("lastError") or "")

        if (not has_plotly or not has_plot) and not self._fallback_to_embedded_js:
            self._fallback_to_embedded_js = True
            self._load_figure(embed_js=True)
            return

        if has_plotly and has_plot and trace_count > 0:
            self._is_health_check_pending = False
            self._has_data = True
            return

        if self._health_check_attempt < HEALTH_CHECK_MAX_ATTEMPTS:
            self._health_check_attempt += 1
            QTimer.singleShot(HEALTH_CHECK_RETRY_MS, self._run_health_check)
            return

        self._is_health_check_pending = False
        self._has_data = False
        detail = f" Детали: {last_error}" if last_error else ""
        self.render_error.emit(
            "График не инициализировался. Проверьте установленный Qt WebEngine и Plotly."
            + detail
        )

    @staticmethod
    def _plotly_config() -> dict[str, object]:
        return {
            "responsive": True,
            "displaylogo": False,
            "toImageButtonOptions": {"format": "png", "filename": "chart"},
        }

    @staticmethod
    def _resolve_local_plotly_js_url() -> Optional[str]:
        if plotly is None:
            return None

        js_path = Path(plotly.__file__).resolve().parent / "package_data" / "plotly.min.js"
        if not js_path.exists():
            return None
        return QUrl.fromLocalFile(str(js_path)).toString()

    @staticmethod
    def _inject_bridge_script(html: str) -> str:
        bridge_script = """
        <script src="qrc:///qtwebchannel/qwebchannel.js"></script>
        <script>
        (function() {
          window.__plotlyLastError = null;
          window.addEventListener('error', function(event) {
            var message = event && event.message ? event.message : 'javascript error';
            window.__plotlyLastError = message;
          });
          window.addEventListener('unhandledrejection', function(event) {
            var reason = event && event.reason ? String(event.reason) : 'promise rejection';
            window.__plotlyLastError = reason;
          });

          function normalizeXToSeconds(xValue, axisType) {
            if (xValue === undefined || xValue === null) return null;
            if (xValue instanceof Date) {
              return String(xValue.getTime() / 1000.0);
            }
            if (typeof xValue === 'number') {
              if (axisType === 'date' && xValue > 100000000000) {
                return String(xValue / 1000.0);
              }
              return String(xValue);
            }
            var text = String(xValue).trim();
            if (!text) return null;
            var parsedMs = Date.parse(text);
            if (!Number.isNaN(parsedMs)) {
              return String(parsedMs / 1000.0);
            }
            return text;
          }

          function resolveXFromPointer(plot, event) {
            if (!plot || !event || !plot._fullLayout) return null;
            var xaxis = plot._fullLayout.xaxis;
            var yaxis = plot._fullLayout.yaxis;
            if (!xaxis || !yaxis || typeof xaxis.p2d !== 'function') return null;

            var rect = plot.getBoundingClientRect();
            var xPix = event.clientX - rect.left - xaxis._offset;
            var yPix = event.clientY - rect.top - yaxis._offset;
            if (!Number.isFinite(xPix) || !Number.isFinite(yPix)) return null;
            if (xPix < 0 || xPix > xaxis._length) return null;
            if (yPix < 0 || yPix > yaxis._length) return null;

            var xValue = xaxis.p2d(xPix);
            return normalizeXToSeconds(xValue, xaxis.type || 'linear');
          }

          function attachClickHandler(plot, bridge) {
            if (!plot || !bridge || !plot.on) return;
            if (plot.__clickBridgeBound) return;
            plot.__clickBridgeBound = true;
            var lastSentValue = null;
            var lastSentAt = 0;

            function emitX(xValue, axisType) {
              var prepared = normalizeXToSeconds(xValue, axisType || 'linear');
              if (!prepared) return;
              var now = Date.now();
              if (prepared === lastSentValue && (now - lastSentAt) < 120) {
                return;
              }
              lastSentValue = prepared;
              lastSentAt = now;
              bridge.onPlotClicked(prepared);
            }

            plot.on('plotly_click', function(ev) {
              if (!ev || !ev.points || !ev.points.length) return;
              var xVal = ev.points[0].x;
              if (xVal === undefined || xVal === null) return;
              var axisType = (ev.points[0].xaxis && ev.points[0].xaxis.type) || 'date';
              emitX(xVal, axisType);
            });

            plot.addEventListener('click', function(event) {
              var xFromPointer = resolveXFromPointer(plot, event);
              if (!xFromPointer) return;
              emitX(xFromPointer, 'linear');
            }, true);
          }

          function setupWebChannel() {
            if (typeof QWebChannel === 'undefined' || typeof qt === 'undefined') {
              return;
            }
            new QWebChannel(qt.webChannelTransport, function(channel) {
              var bridge = channel.objects.bridge;
              var tries = 0;
              function waitPlot() {
                var plot = document.querySelector('.plotly-graph-div');
                if (plot) {
                  attachClickHandler(plot, bridge);
                  return;
                }
                tries += 1;
                if (tries < 40) {
                  setTimeout(waitPlot, 150);
                }
              }
              waitPlot();
            });
          }

          if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', setupWebChannel);
          } else {
            setupWebChannel();
          }
        })();
        </script>
        """
        if "</body>" in html:
            return html.replace("</body>", bridge_script + "</body>")
        return html + bridge_script


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Построение графиков")
        self.resize(1520, 900)

        self._service: Optional[SQLiteService] = None
        self._time_bounds: Optional[tuple[float, float]] = None
        self._last_traces_payload: list[dict] = []
        self._chart_cache: OrderedDict[tuple, dict[str, object]] = OrderedDict()
        self._build_thread: Optional[ChartBuildThread] = None
        self._build_request_seq = 0
        self._active_request_id = 0
        self._queued_build_args: Optional[tuple[list[tuple[str, str]], float, float]] = None

        self._build_ui()
        self._apply_styles()
        self._set_data_controls_enabled(False)
        self._set_placeholder_chart()

        if not PLOTLY_AVAILABLE:
            self.statusBar().showMessage(
                "Plotly не установлен. Выполните: pip install -r requirements.txt"
            )

    def _build_ui(self) -> None:
        open_action = QAction("Открыть базу...", self)
        open_action.triggered.connect(self._open_database_dialog)

        save_image_action = QAction("Сохранить график как изображение...", self)
        save_image_action.triggered.connect(self._save_chart_image_dialog)

        file_menu = self.menuBar().addMenu("Файл")
        file_menu.addAction(open_action)
        file_menu.addAction(save_image_action)

        central_widget = QWidget(self)
        central_widget.setObjectName("centralRoot")
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        controls_container = QWidget(self)
        controls_container.setObjectName("controlsPanel")
        controls_layout = QVBoxLayout(controls_container)
        controls_layout.setContentsMargins(14, 14, 14, 14)
        controls_layout.setSpacing(8)

        panel_title = QLabel("Панель управления")
        panel_title.setObjectName("panelTitle")
        controls_layout.addWidget(panel_title)

        self.open_db_button = QPushButton("Открыть .db файл")
        self.open_db_button.setProperty("variant", "accent")
        self.open_db_button.clicked.connect(self._open_database_dialog)
        controls_layout.addWidget(self.open_db_button)

        controls_layout.addWidget(self._make_section_label("База данных"))
        self.path_label = QLabel("Файл не выбран")
        self.path_label.setObjectName("pathLabel")
        self.path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.path_label.setWordWrap(True)
        controls_layout.addWidget(self.path_label)

        controls_layout.addWidget(self._make_section_label("Ось X (фиксированная)"))
        self.x_label = QLabel(f"{TIME_COLUMN} (Unix time, секунды)")
        self.x_label.setWordWrap(True)
        self.x_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        controls_layout.addWidget(self.x_label)

        controls_layout.addWidget(self._make_section_label("Начало"))
        self.start_dt_edit = QDateTimeEdit()
        self.start_dt_edit.setCalendarPopup(True)
        self.start_dt_edit.setDisplayFormat("dd.MM.yyyy HH:mm:ss.zzz")
        self.start_dt_edit.editingFinished.connect(self._on_left_filters_changed)
        controls_layout.addWidget(self.start_dt_edit)

        controls_layout.addWidget(self._make_section_label("Конец"))
        self.end_dt_edit = QDateTimeEdit()
        self.end_dt_edit.setCalendarPopup(True)
        self.end_dt_edit.setDisplayFormat("dd.MM.yyyy HH:mm:ss.zzz")
        self.end_dt_edit.editingFinished.connect(self._on_left_filters_changed)
        controls_layout.addWidget(self.end_dt_edit)

        controls_layout.addWidget(self._make_section_label("День"))
        day_row = QWidget(self)
        day_row_layout = QHBoxLayout(day_row)
        day_row_layout.setContentsMargins(0, 0, 0, 0)
        day_row_layout.setSpacing(6)
        self.day_edit = QDateEdit()
        self.day_edit.setCalendarPopup(True)
        self.day_edit.setDisplayFormat("dd.MM.yyyy")
        day_row_layout.addWidget(self.day_edit, 1)

        self.apply_day_button = QPushButton("Выбрать день")
        self.apply_day_button.setProperty("variant", "ghost")
        self.apply_day_button.clicked.connect(self._apply_selected_day)
        day_row_layout.addWidget(self.apply_day_button)
        controls_layout.addWidget(day_row)

        self.full_range_button = QPushButton("Полный диапазон")
        self.full_range_button.setProperty("variant", "ghost")
        self.full_range_button.clicked.connect(self._use_full_time_range)
        controls_layout.addWidget(self.full_range_button)

        rows_row = QWidget(self)
        rows_row_layout = QHBoxLayout(rows_row)
        rows_row_layout.setContentsMargins(0, 0, 0, 0)
        rows_row_layout.setSpacing(6)
        rows_row_layout.addWidget(QLabel("Строк на канал"))
        self.limit_spin = QSpinBox()
        self.limit_spin.setRange(10, 1_000_000)
        self.limit_spin.setSingleStep(100)
        self.limit_spin.setValue(4511)
        self.limit_spin.valueChanged.connect(self._on_left_filters_changed)
        rows_row_layout.addWidget(self.limit_spin, 1)
        controls_layout.addWidget(rows_row)

        controls_layout.addWidget(self._make_section_label("Каналы Y"))
        self.channels_list = QListWidget(self)
        self.channels_list.setSelectionMode(QAbstractItemView.NoSelection)
        self.channels_list.setMinimumHeight(180)
        self.channels_list.itemChanged.connect(self._on_channels_selection_changed)
        controls_layout.addWidget(self.channels_list, 1)

        self.channels_info_label = QLabel("Выбрано каналов: 0")
        self.channels_info_label.setObjectName("hintLabel")
        controls_layout.addWidget(self.channels_info_label)

        self.show_y_axes_checkbox = QCheckBox("Показывать оси Y")
        self.show_y_axes_checkbox.setChecked(True)
        self.show_y_axes_checkbox.stateChanged.connect(self._on_y_axes_visibility_changed)
        controls_layout.addWidget(self.show_y_axes_checkbox)

        channels_action_row = QWidget(self)
        channels_action_layout = QHBoxLayout(channels_action_row)
        channels_action_layout.setContentsMargins(0, 0, 0, 0)
        channels_action_layout.setSpacing(6)
        self.select_all_button = QPushButton("Выбрать все")
        self.select_all_button.setProperty("variant", "ghost")
        self.select_all_button.clicked.connect(self._select_all_channels)
        channels_action_layout.addWidget(self.select_all_button)

        self.clear_selection_button = QPushButton("Очистить")
        self.clear_selection_button.setProperty("variant", "ghost")
        self.clear_selection_button.clicked.connect(self._clear_channel_selection)
        channels_action_layout.addWidget(self.clear_selection_button)
        controls_layout.addWidget(channels_action_row)

        build_action_row = QWidget(self)
        build_action_layout = QHBoxLayout(build_action_row)
        build_action_layout.setContentsMargins(0, 0, 0, 0)
        build_action_layout.setSpacing(6)
        self.plot_selected_button = QPushButton("Построить выбранные")
        self.plot_selected_button.setProperty("variant", "accent")
        self.plot_selected_button.clicked.connect(self._build_selected_chart)
        build_action_layout.addWidget(self.plot_selected_button)

        self.plot_all_button = QPushButton("Построить все")
        self.plot_all_button.setProperty("variant", "accent")
        self.plot_all_button.clicked.connect(self._build_all_channels)
        build_action_layout.addWidget(self.plot_all_button)
        controls_layout.addWidget(build_action_row)

        self.reset_zoom_button = QPushButton("Сбросить масштаб")
        self.reset_zoom_button.setProperty("variant", "ghost")
        self.reset_zoom_button.clicked.connect(self._reset_zoom)
        controls_layout.addWidget(self.reset_zoom_button)

        self.save_image_button = QPushButton("Сохранить график как изображение")
        self.save_image_button.setProperty("variant", "accent")
        self.save_image_button.clicked.connect(self._save_chart_image_dialog)
        controls_layout.addWidget(self.save_image_button)

        controls_layout.addWidget(self._make_section_label("Значения по клику"))
        self.clicked_time_label = QLabel("Время X: —")
        self.clicked_time_label.setObjectName("clickedTime")
        self.clicked_time_label.setWordWrap(True)
        controls_layout.addWidget(self.clicked_time_label)

        self.clicked_values_list = QListWidget(self)
        self.clicked_values_list.setSelectionMode(QAbstractItemView.NoSelection)
        self.clicked_values_list.setMinimumHeight(160)
        controls_layout.addWidget(self.clicked_values_list)

        self.interaction_hint_label = QLabel(
            "Клик по графику: определяется X и показываются значения всех выбранных каналов Y "
            "в ближайшей точке времени."
        )
        self.interaction_hint_label.setObjectName("hintLabel")
        self.interaction_hint_label.setWordWrap(True)
        controls_layout.addWidget(self.interaction_hint_label)
        controls_layout.addStretch(1)

        self.plot_view = PlotlyChartView(self)
        self.plot_view.setObjectName("plotCanvas")
        self.plot_view.point_clicked.connect(self._on_plot_point_clicked)
        self.plot_view.render_error.connect(self._on_plot_render_error)

        chart_container = QWidget(self)
        chart_container.setObjectName("chartPanel")
        chart_layout = QVBoxLayout(chart_container)
        chart_layout.setContentsMargins(10, 10, 10, 10)
        chart_layout.setSpacing(0)
        chart_layout.addWidget(self.plot_view, 1)

        controls_scroll = QScrollArea(self)
        controls_scroll.setObjectName("controlsScroll")
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        controls_scroll.setFrameShape(QFrame.NoFrame)
        controls_scroll.setMinimumWidth(360)
        controls_scroll.setMaximumWidth(430)
        controls_scroll.setWidget(controls_container)

        main_layout.addWidget(controls_scroll)
        main_layout.addWidget(chart_container, 1)

        self.setCentralWidget(central_widget)
        self.statusBar().showMessage("Откройте базу данных для начала работы.")

    def _make_section_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setProperty("section", True)
        return label

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #e9eff6;
                color: #0f172a;
                font-family: "Avenir Next", "Segoe UI", "Helvetica Neue", sans-serif;
                font-size: 13px;
            }
            QMenuBar {
                background-color: #0f172a;
                color: #e2e8f0;
                border-bottom: 1px solid #1e293b;
            }
            QMenuBar::item {
                background: transparent;
                padding: 6px 10px;
                margin: 2px;
                border-radius: 8px;
            }
            QMenuBar::item:selected {
                background-color: #1e293b;
            }
            QMenu {
                background-color: #ffffff;
                border: 1px solid #cbd5e1;
                padding: 6px;
            }
            QMenu::item {
                padding: 6px 12px;
                border-radius: 6px;
            }
            QMenu::item:selected {
                background-color: #e2e8f0;
            }
            QWidget#controlsPanel {
                background: qlineargradient(
                    x1: 0,
                    y1: 0,
                    x2: 0,
                    y2: 1,
                    stop: 0 #ffffff,
                    stop: 1 #f6f9ff
                );
                border: 1px solid #d3ddeb;
                border-radius: 16px;
            }
            QScrollArea#controlsScroll {
                background: transparent;
                border: none;
            }
            QScrollArea#controlsScroll > QWidget > QWidget {
                background: transparent;
            }
            QWidget#chartPanel {
                background-color: #ffffff;
                border: 1px solid #d3ddeb;
                border-radius: 16px;
            }
            QLabel#panelTitle {
                font-size: 19px;
                font-weight: 700;
                color: #0f172a;
                padding: 4px 2px 8px 2px;
            }
            QLabel[section="true"] {
                font-weight: 700;
                color: #334155;
                margin-top: 6px;
            }
            QLabel#pathLabel,
            QLabel#clickedTime,
            QLabel#hintLabel {
                background: #f8fbff;
                border: 1px solid #dbe5f2;
                border-radius: 10px;
                padding: 8px;
                color: #334155;
            }
            QPushButton {
                background-color: #f1f5f9;
                color: #0f172a;
                border: 1px solid #cbd5e1;
                border-radius: 10px;
                padding: 8px 12px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #e6edf5;
            }
            QPushButton:pressed {
                background-color: #d8e3ef;
            }
            QPushButton[variant="accent"] {
                background-color: #0f766e;
                border: 1px solid #0f766e;
                color: #ecfeff;
            }
            QPushButton[variant="accent"]:hover {
                background-color: #0d675f;
            }
            QPushButton[variant="accent"]:pressed {
                background-color: #0b5953;
            }
            QPushButton[variant="ghost"] {
                background-color: #ffffff;
                color: #1e293b;
            }
            QDateEdit,
            QDateTimeEdit,
            QSpinBox,
            QListWidget {
                background-color: #ffffff;
                border: 1px solid #cbd5e1;
                border-radius: 10px;
                padding: 6px;
                color: #0f172a;
            }
            QDateEdit:focus,
            QDateTimeEdit:focus,
            QSpinBox:focus,
            QListWidget:focus {
                border: 1px solid #0f766e;
            }
            QScrollBar:vertical {
                background: #f1f5f9;
                width: 12px;
                margin: 2px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background: #cbd5e1;
                min-height: 24px;
                border-radius: 6px;
            }
            QCheckBox {
                color: #0f172a;
                spacing: 6px;
                font-weight: 600;
                padding: 4px 2px 2px 2px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border-radius: 4px;
                border: 1px solid #94a3b8;
                background: #ffffff;
            }
            QCheckBox::indicator:checked {
                background: #0f766e;
                border: 1px solid #0f766e;
            }
            QStatusBar {
                background-color: #0f172a;
                color: #e2e8f0;
                border-top: 1px solid #1e293b;
            }
            """
        )

    def _set_data_controls_enabled(self, enabled: bool) -> None:
        actual_enabled = enabled and PLOTLY_AVAILABLE
        self.start_dt_edit.setEnabled(actual_enabled)
        self.end_dt_edit.setEnabled(actual_enabled)
        self.day_edit.setEnabled(actual_enabled)
        self.apply_day_button.setEnabled(actual_enabled)
        self.full_range_button.setEnabled(actual_enabled)
        self.limit_spin.setEnabled(actual_enabled)
        self.select_all_button.setEnabled(actual_enabled)
        self.clear_selection_button.setEnabled(actual_enabled)
        selected_count = len(self._selected_channels()) if actual_enabled else 0
        self.plot_selected_button.setEnabled(actual_enabled and selected_count > 0)
        self.plot_all_button.setEnabled(actual_enabled)
        self.reset_zoom_button.setEnabled(actual_enabled)
        self.save_image_button.setEnabled(actual_enabled)
        self.channels_list.setEnabled(actual_enabled)
        self.show_y_axes_checkbox.setEnabled(actual_enabled)
        self.clicked_values_list.setEnabled(actual_enabled)
        self.channels_info_label.setText(f"Выбрано каналов: {selected_count}")

    def _open_database_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Выбор SQLite файла",
            str(Path.home()),
            "SQLite файлы (*.db *.sqlite *.sqlite3);;Все файлы (*)",
        )
        if not path:
            return
        self._load_database(path)

    def _load_database(self, path: str) -> None:
        if not PLOTLY_AVAILABLE:
            self._show_error("Plotly не установлен. Выполните: pip install -r requirements.txt")
            return

        service = SQLiteService(path)
        try:
            if not service.table_exists(DATA_TABLE):
                self._show_error(f"Таблица '{DATA_TABLE}' не найдена в выбранной базе.")
                self._set_data_controls_enabled(False)
                self._set_placeholder_chart()
                return

            columns = service.list_columns(DATA_TABLE)
            time_bounds = service.fetch_time_bounds(DATA_TABLE, TIME_COLUMN)
        except sqlite3.Error as error:
            self._show_error(f"Ошибка при открытии базы:\n{error}")
            self._set_data_controls_enabled(False)
            self._set_placeholder_chart()
            return

        column_names = {column.name for column in columns}
        if TIME_COLUMN not in column_names:
            self._show_error(
                f"В таблице '{DATA_TABLE}' отсутствует обязательный столбец '{TIME_COLUMN}'."
            )
            self._set_data_controls_enabled(False)
            self._set_placeholder_chart()
            return

        if not time_bounds:
            self._show_error(f"В таблице '{DATA_TABLE}' нет корректных значений времени.")
            self._set_data_controls_enabled(False)
            self._set_placeholder_chart()
            return

        available_channels = [
            (column_name, label)
            for column_name, label in Y_CHANNELS
            if column_name in column_names
        ]
        if not available_channels:
            self._show_error("В таблице 'data' не найдены поддерживаемые каналы Y.")
            self._set_data_controls_enabled(False)
            self._set_placeholder_chart()
            return

        self._service = service
        self._time_bounds = time_bounds
        self._chart_cache.clear()
        self._queued_build_args = None
        self._active_request_id = 0
        self.path_label.setText(path)

        self._fill_channel_list(available_channels)
        self._set_time_controls_from_bounds(time_bounds)
        self._set_data_controls_enabled(True)

        index_hint = ""
        try:
            index_name = service.ensure_time_index(DATA_TABLE, TIME_COLUMN)
            index_hint = f" Индекс времени: {index_name}."
        except sqlite3.Error:
            index_hint = " Индекс времени не создан (read-only база)."

        self._build_selected_chart()
        self.statusBar().showMessage("База успешно загружена." + index_hint)

    def _fill_channel_list(self, channels: list[tuple[str, str]]) -> None:
        self.channels_list.clear()
        blocker = QSignalBlocker(self.channels_list)
        for index, (column_name, label) in enumerate(channels):
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, column_name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked if index == 0 else Qt.CheckState.Unchecked
            )
            self.channels_list.addItem(item)
        del blocker
        self._on_channels_selection_changed()

    def _set_time_controls_from_bounds(self, bounds: tuple[float, float]) -> None:
        min_ts, max_ts = bounds
        min_dt = QDateTime.fromMSecsSinceEpoch(int(round(min_ts * 1000.0)))
        max_dt = QDateTime.fromMSecsSinceEpoch(int(round(max_ts * 1000.0)))

        self.start_dt_edit.setMinimumDateTime(min_dt)
        self.start_dt_edit.setMaximumDateTime(max_dt)
        self.end_dt_edit.setMinimumDateTime(min_dt)
        self.end_dt_edit.setMaximumDateTime(max_dt)
        self.start_dt_edit.setDateTime(min_dt)
        self.end_dt_edit.setDateTime(max_dt)

        self.day_edit.setMinimumDate(min_dt.date())
        self.day_edit.setMaximumDate(max_dt.date())
        self.day_edit.setDate(min_dt.date())

    def _build_all_channels(self) -> None:
        had_chart = self.plot_view.has_data()
        self._select_all_channels()
        if not had_chart:
            self._build_selected_chart()

    def _build_selected_chart(self) -> None:
        if not self._service or not PLOTLY_AVAILABLE:
            return

        selected_channels = self._selected_channels()
        if not selected_channels:
            self._show_error("Выберите хотя бы один канал Y.")
            return

        time_range = self._selected_time_range()
        if not time_range:
            return
        start_ts, end_ts = time_range
        self._request_chart_build(selected_channels, start_ts, end_ts)

    def _request_chart_build(
        self,
        selected_channels: list[tuple[str, str]],
        start_ts: float,
        end_ts: float,
    ) -> None:
        if not self._service:
            return

        if self._build_thread and self._build_thread.isRunning():
            self._queued_build_args = (selected_channels, start_ts, end_ts)
            self.statusBar().showMessage("Идет построение графика, обновление поставлено в очередь...")
            return

        cache_key = self._build_cache_key(selected_channels, start_ts, end_ts)
        cached = self._cache_get(cache_key)
        if cached is not None:
            traces_payload = cached["traces_payload"]  # type: ignore[assignment]
            points_total = int(cached["points_total"])  # type: ignore[arg-type]
            self._apply_traces_payload(
                traces_payload=traces_payload,
                selected_count=len(selected_channels),
                start_ts=start_ts,
                end_ts=end_ts,
                points_total=points_total,
                from_cache=True,
            )
            return

        self._build_request_seq += 1
        request_id = self._build_request_seq
        self._active_request_id = request_id
        self.statusBar().showMessage("Подготовка графика...")

        self._build_thread = ChartBuildThread(
            request_id=request_id,
            db_path=self._service.db_path,
            selected_channels=selected_channels,
            limit=self.limit_spin.value(),
            start_ts=start_ts,
            end_ts=end_ts,
            parent=self,
        )
        self._build_thread.build_succeeded.connect(
            lambda req_id, traces_payload, points_total: self._on_build_succeeded(
                req_id=req_id,
                traces_payload=traces_payload,
                points_total=points_total,
                selected_channels=selected_channels,
                start_ts=start_ts,
                end_ts=end_ts,
                cache_key=cache_key,
            )
        )
        self._build_thread.build_failed.connect(self._on_build_failed)
        self._build_thread.finished.connect(self._on_build_thread_finished)
        self._build_thread.start()

    def _on_build_succeeded(
        self,
        req_id: int,
        traces_payload: list[dict],
        points_total: int,
        selected_channels: list[tuple[str, str]],
        start_ts: float,
        end_ts: float,
        cache_key: tuple,
    ) -> None:
        if req_id != self._active_request_id:
            return

        self._cache_set(
            cache_key,
            {"traces_payload": traces_payload, "points_total": points_total},
        )
        self._apply_traces_payload(
            traces_payload=traces_payload,
            selected_count=len(selected_channels),
            start_ts=start_ts,
            end_ts=end_ts,
            points_total=points_total,
            from_cache=False,
        )

    def _on_build_failed(self, req_id: int, message: str) -> None:
        if req_id != self._active_request_id:
            return
        self._show_error(message)

    def _on_build_thread_finished(self) -> None:
        if self._build_thread:
            self._build_thread.deleteLater()
            self._build_thread = None
        self._process_queued_build()

    def _process_queued_build(self) -> None:
        if not self._queued_build_args:
            return
        queued = self._queued_build_args
        self._queued_build_args = None
        selected_channels, start_ts, end_ts = queued
        self._request_chart_build(selected_channels, start_ts, end_ts)

    def _apply_traces_payload(
        self,
        traces_payload: list[dict],
        selected_count: int,
        start_ts: float,
        end_ts: float,
        points_total: int,
        from_cache: bool,
    ) -> None:
        if not traces_payload:
            self._last_traces_payload = []
            self.plot_view.set_placeholder("Нет данных в выбранном диапазоне времени.")
            self.clicked_time_label.setText("Время X: —")
            self.clicked_values_list.clear()
            self.statusBar().showMessage("Нет точек для отображения в выбранном диапазоне.")
            return

        figure = self._build_plotly_figure(
            traces_payload=traces_payload,
            selected_count=selected_count,
            start_ts=start_ts,
            end_ts=end_ts,
            show_y_axes=self.show_y_axes_checkbox.isChecked(),
        )
        self.plot_view.set_figure(figure)
        self._last_traces_payload = traces_payload
        self.clicked_time_label.setText("Время X: —")
        self.clicked_values_list.clear()

        self.statusBar().showMessage(
            f"Построен график: {len(traces_payload)}/{selected_count} каналов, "
            f"{points_total} точек."
            + (" (кэш)" if from_cache else "")
        )

    def _build_cache_key(
        self,
        selected_channels: list[tuple[str, str]],
        start_ts: float,
        end_ts: float,
    ) -> tuple:
        if not self._service:
            return ()
        channel_key = tuple(column for column, _ in selected_channels)
        return (
            self._service.db_path,
            int(round(start_ts * 1000.0)),
            int(round(end_ts * 1000.0)),
            channel_key,
            int(self.limit_spin.value()),
        )

    def _cache_get(self, key: tuple) -> Optional[dict[str, object]]:
        if not key:
            return None
        payload = self._chart_cache.get(key)
        if payload is None:
            return None
        self._chart_cache.move_to_end(key)
        return payload

    def _cache_set(self, key: tuple, value: dict[str, object]) -> None:
        if not key:
            return
        self._chart_cache[key] = value
        self._chart_cache.move_to_end(key)
        while len(self._chart_cache) > CACHE_MAX_ITEMS:
            self._chart_cache.popitem(last=False)

    def _build_plotly_figure(
        self,
        traces_payload: list[dict],
        selected_count: int,
        start_ts: float,
        end_ts: float,
        show_y_axes: bool,
    ):
        assert go is not None
        figure = go.Figure()

        for index, payload in enumerate(traces_payload):
            yaxis_name = "y" if index == 0 else f"y{index + 1}"
            figure.add_trace(
                go.Scatter(
                    x=payload["x_values"],
                    y=payload["y_values"],
                    mode="lines",
                    name=payload["name"],
                    line={"color": payload["color"], "width": 2},
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
                "range": [payload["y_min"], payload["y_max"]],
                "zeroline": False,
            }
            if show_y_axes:
                base_cfg.update(
                    {
                        "title": {"text": payload["name"], "font": {"color": payload["color"]}},
                        "tickfont": {"color": payload["color"]},
                    }
                )
            else:
                base_cfg.update(
                    {
                        "visible": False,
                        "showgrid": False,
                        "showticklabels": False,
                    }
                )

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
                f"Каналы: {selected_count} | "
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

    def _on_plot_point_clicked(self, x_value: str) -> None:
        if not self._last_traces_payload:
            return

        x_ts = self._parse_plotly_x_to_timestamp(x_value)
        if x_ts is None:
            self.clicked_time_label.setText(f"Время X: не распознано ({x_value})")
            self.clicked_values_list.clear()
            return

        clicked_dt = datetime.fromtimestamp(x_ts)
        self.clicked_time_label.setText(
            f"Время X: {clicked_dt.strftime('%d.%m.%Y %H:%M:%S.%f')[:-3]}"
        )
        self.clicked_values_list.clear()

        for payload in self._last_traces_payload:
            ts_values = payload["ts_values"]
            y_values = payload["y_values"]
            if not ts_values:
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
            nearest_y = y_values[best_idx]
            clean_name = self._strip_type_suffix(payload["name"])
            self.clicked_values_list.addItem(
                f"{clean_name}: {self._format_number(nearest_y)}"
            )

    def _on_channels_selection_changed(self, _item: Optional[QListWidgetItem] = None) -> None:
        selected_count = len(self._selected_channels())
        self.channels_info_label.setText(f"Выбрано каналов: {selected_count}")
        can_build = self.channels_list.isEnabled() and PLOTLY_AVAILABLE and selected_count > 0
        self.plot_selected_button.setEnabled(can_build)

        if not self._service or not self.plot_view.has_data():
            return

        if selected_count == 0:
            self._build_request_seq += 1
            self._active_request_id = self._build_request_seq
            self._queued_build_args = None
            self._last_traces_payload = []
            self.plot_view.set_placeholder("Выберите хотя бы один канал Y.")
            self.clicked_time_label.setText("Время X: —")
            self.clicked_values_list.clear()
            self.statusBar().showMessage("Каналы не выбраны.")
            return

        self._build_selected_chart()

    def _selected_channels(self) -> list[tuple[str, str]]:
        channels: list[tuple[str, str]] = []
        for index in range(self.channels_list.count()):
            item = self.channels_list.item(index)
            if item.checkState() == Qt.CheckState.Checked:
                column_name = item.data(Qt.ItemDataRole.UserRole)
                if column_name:
                    channels.append((str(column_name), item.text()))
        return channels

    def _selected_time_range(self) -> Optional[tuple[float, float]]:
        start_dt = self.start_dt_edit.dateTime()
        end_dt = self.end_dt_edit.dateTime()
        if start_dt > end_dt:
            self._show_error("Время начала должно быть меньше или равно времени конца.")
            return None
        start_ts = start_dt.toMSecsSinceEpoch() / 1000.0
        end_ts = end_dt.toMSecsSinceEpoch() / 1000.0
        return start_ts, end_ts

    def _apply_selected_day(self) -> None:
        if not self._time_bounds:
            return
        selected_date = self.day_edit.date()
        day_start = selected_date.startOfDay()
        day_end = day_start.addDays(1).addMSecs(-1)

        min_dt = self.start_dt_edit.minimumDateTime()
        max_dt = self.start_dt_edit.maximumDateTime()
        if day_end < min_dt or day_start > max_dt:
            self._show_error("Выбранный день вне диапазона данных.")
            return

        if day_start < min_dt:
            day_start = min_dt
        if day_end > max_dt:
            day_end = max_dt

        self.start_dt_edit.setDateTime(day_start)
        self.end_dt_edit.setDateTime(day_end)
        if self._service and self.plot_view.has_data() and self._selected_channels():
            self._build_selected_chart()

    def _use_full_time_range(self) -> None:
        min_dt = self.start_dt_edit.minimumDateTime()
        max_dt = self.start_dt_edit.maximumDateTime()
        self.start_dt_edit.setDateTime(min_dt)
        self.end_dt_edit.setDateTime(max_dt)
        if self._service and self.plot_view.has_data() and self._selected_channels():
            self._build_selected_chart()

    def _select_all_channels(self) -> None:
        blocker = QSignalBlocker(self.channels_list)
        for index in range(self.channels_list.count()):
            item = self.channels_list.item(index)
            item.setCheckState(Qt.CheckState.Checked)
        del blocker
        self._on_channels_selection_changed()

    def _clear_channel_selection(self) -> None:
        blocker = QSignalBlocker(self.channels_list)
        for index in range(self.channels_list.count()):
            item = self.channels_list.item(index)
            item.setCheckState(Qt.CheckState.Unchecked)
        del blocker
        self._on_channels_selection_changed()

    def _reset_zoom(self) -> None:
        self.plot_view.reset_view()

    def _on_left_filters_changed(self, _value: object = None) -> None:
        if not self._service or not self.plot_view.has_data():
            return
        if not self._selected_channels():
            return
        time_range = self._selected_time_range()
        if not time_range:
            return
        self._build_selected_chart()

    def _on_y_axes_visibility_changed(self, _state: int) -> None:
        if not self._service or not self._last_traces_payload:
            return
        selected_channels = self._selected_channels()
        if not selected_channels:
            return
        time_range = self._selected_time_range()
        if not time_range:
            return
        start_ts, end_ts = time_range
        figure = self._build_plotly_figure(
            traces_payload=self._last_traces_payload,
            selected_count=len(selected_channels),
            start_ts=start_ts,
            end_ts=end_ts,
            show_y_axes=self.show_y_axes_checkbox.isChecked(),
        )
        self.plot_view.set_figure(figure)

    def _on_plot_render_error(self, message: str) -> None:
        self.statusBar().showMessage(message, 10000)

    def _set_placeholder_chart(self, title: str = "Откройте базу данных и постройте график.") -> None:
        if not PLOTLY_AVAILABLE:
            self.plot_view.set_placeholder(
                "Plotly не установлен. Выполните: pip install -r requirements.txt"
            )
            return
        self.plot_view.set_placeholder(title)

    def _save_chart_image_dialog(self) -> None:
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Сохранить график как изображение",
            str(Path.home() / "chart.png"),
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;WebP (*.webp);;BMP (*.bmp)",
        )
        if not path:
            return
        path = self._ensure_image_extension(path, selected_filter)

        try:
            self._export_chart_to_image(path)
        except Exception as error:  # pragma: no cover - runtime/UI path
            self._show_error(f"Ошибка сохранения изображения:\n{error}")
            return

        self.statusBar().showMessage(f"График сохранен как изображение: {path}")

    def _export_chart_to_image(self, path: str) -> None:
        if not self.plot_view.has_data():
            raise ValueError("Нет данных графика для экспорта.")

        pixmap = self.plot_view.grab_chart()
        if pixmap.isNull():
            raise RuntimeError("Не удалось получить изображение графика.")

        extension = Path(path).suffix.lower().lstrip(".")
        image_format = {
            "png": "PNG",
            "jpg": "JPG",
            "jpeg": "JPG",
            "webp": "WEBP",
            "bmp": "BMP",
        }.get(extension, "PNG")

        if not pixmap.save(path, image_format):
            raise RuntimeError("Qt не смог сохранить изображение в выбранный файл.")

    @staticmethod
    def _ensure_image_extension(path: str, selected_filter: str) -> str:
        if Path(path).suffix:
            return path

        lowered = (selected_filter or "").lower()
        if "jpeg" in lowered or "jpg" in lowered:
            return f"{path}.jpg"
        if "webp" in lowered:
            return f"{path}.webp"
        if "bmp" in lowered:
            return f"{path}.bmp"
        return f"{path}.png"

    def _show_error(self, message: str) -> None:
        QMessageBox.critical(self, "Ошибка", message)
        self.statusBar().showMessage(message.replace("\n", " "), 7000)

    @staticmethod
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

    @staticmethod
    def _format_number(value: float) -> str:
        if abs(value - round(value)) < 1e-9:
            return str(int(round(value)))
        return f"{value:.6f}".rstrip("0").rstrip(".")

    @staticmethod
    def _strip_type_suffix(label: str) -> str:
        text = (label or "").strip()
        if text.endswith(")") and " (" in text:
            return text.rsplit(" (", 1)[0]
        return text
