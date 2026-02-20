@echo off
setlocal

powershell -ExecutionPolicy Bypass -File "%~dp0build_windows.ps1"
if errorlevel 1 (
  echo.
  echo Ошибка сборки.
  exit /b 1
)

echo.
echo Готово: dist\GraphicsWebApp\GraphicsWebApp.exe
exit /b 0
