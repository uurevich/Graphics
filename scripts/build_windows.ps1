$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Resolve-Path (Join-Path $scriptDir "..")
Set-Location $projectRoot

$venvPath = Join-Path $projectRoot ".venv-win"
$pythonExe = Join-Path $venvPath "Scripts\python.exe"

if (-not (Test-Path $pythonExe)) {
    py -3 -m venv $venvPath
}

& $pythonExe -m pip install --upgrade pip
& $pythonExe -m pip install -r "requirements-build.txt"

if (Test-Path "build") {
    Remove-Item -Recurse -Force "build"
}
if (Test-Path "dist") {
    Remove-Item -Recurse -Force "dist"
}
if (Test-Path "GraphicsApp.spec") {
    Remove-Item -Force "GraphicsApp.spec"
}

& $pythonExe -m PyInstaller `
    --noconfirm `
    --clean `
    --windowed `
    --onedir `
    --name "GraphicsApp" `
    --collect-data plotly `
    --collect-submodules PySide6.QtWebEngineCore `
    --collect-submodules PySide6.QtWebEngineWidgets `
    --collect-submodules PySide6.QtWebChannel `
    --collect-binaries PySide6 `
    --collect-data PySide6 `
    "app/main.py"

Write-Host ""
Write-Host "Сборка завершена."
Write-Host "EXE: dist\GraphicsApp\GraphicsApp.exe"
