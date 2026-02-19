import os
import sys
from pathlib import Path


def configure_frozen_qt_runtime() -> None:
    """Configure Qt/WebEngine paths for PyInstaller builds."""
    if not getattr(sys, "frozen", False):
        return

    exe_dir = Path(sys.executable).resolve().parent
    meipass = Path(getattr(sys, "_MEIPASS", exe_dir / "_internal"))

    candidates = [
        meipass / "PySide6",
        exe_dir / "_internal" / "PySide6",
        exe_dir / "PySide6",
    ]
    pyside_dir = next((path for path in candidates if path.exists()), None)
    if pyside_dir is None:
        return

    _set_env_if_exists("QT_PLUGIN_PATH", pyside_dir / "plugins")
    _set_env_if_exists("QML2_IMPORT_PATH", pyside_dir / "qml")
    _set_env_if_exists("QTWEBENGINE_RESOURCES_PATH", pyside_dir / "resources")
    _set_env_if_exists(
        "QTWEBENGINE_LOCALES_PATH",
        pyside_dir / "translations" / "qtwebengine_locales",
    )

    webengine_process = pyside_dir / "QtWebEngineProcess.exe"
    if not webengine_process.exists():
        webengine_process = pyside_dir / "QtWebEngineProcess"
    _set_env_if_exists("QTWEBENGINEPROCESS_PATH", webengine_process)

    if sys.platform.startswith("win"):
        # Software OpenGL + GPU-off flags improve compatibility on older Windows setups.
        os.environ.setdefault("QT_OPENGL", "software")
        os.environ.setdefault("QTWEBENGINE_DISABLE_SANDBOX", "1")
        os.environ.setdefault(
            "QTWEBENGINE_CHROMIUM_FLAGS",
            "--disable-gpu --disable-gpu-compositing",
        )


def _set_env_if_exists(name: str, path: Path) -> None:
    if name in os.environ:
        return
    if path.exists():
        os.environ[name] = str(path)
