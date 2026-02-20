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
if (Test-Path "GraphicsWebApp.spec") {
    Remove-Item -Force "GraphicsWebApp.spec"
}

& $pythonExe -m PyInstaller `
    --noconfirm `
    --clean `
    --windowed `
    --onedir `
    --name "GraphicsWebApp" `
    --add-data "assets;assets" `
    --collect-all dash `
    --collect-data plotly `
    --collect-all flask `
    --collect-all werkzeug `
    --collect-all jinja2 `
    "app/web_app.py"

Write-Host ""
Write-Host "Сборка завершена."
Write-Host "EXE: dist\GraphicsWebApp\GraphicsWebApp.exe"
