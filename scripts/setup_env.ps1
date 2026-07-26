<#
.SYNOPSIS
    Create the MC-MOT development environment on Windows.

.DESCRIPTION
    Builds a Python 3.10 virtual environment, installs the project in editable
    mode with dev tooling, and optionally installs the CUDA build of PyTorch.

    PyTorch is installed separately because the Windows CUDA wheels live on
    PyTorch's own package index, which cannot be expressed as a dependency in
    pyproject.toml. cu121 is selected deliberately: driver 537.70 on this machine
    supports CUDA up to 12.2, and CUDA minor-version compatibility makes the 12.1
    build the safe choice. See README.md for the full setup walkthrough.

.EXAMPLE
    .\scripts\setup_env.ps1 -WithStorage
    .\scripts\setup_env.ps1 -WithStorage -WithTorch -WithMl
#>
[CmdletBinding()]
param(
    [switch]$WithStorage, # phase 2: SQLAlchemy + faiss-cpu (small, no torch)
    [switch]$WithTorch,   # phase 3+: torch + torchvision, CUDA 12.1 (~2.3 GB)
    [switch]$WithMl,      # phase 3+: ultralytics + opencv (requires -WithTorch)
    [switch]$Recreate     # delete an existing .venv first
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$venv = Join-Path $root '.venv'
$python = Join-Path $venv 'Scripts\python.exe'

if ($Recreate -and (Test-Path $venv)) {
    Write-Host '==> Removing existing virtual environment' -ForegroundColor Yellow
    Remove-Item -Recurse -Force $venv
}

if (-not (Test-Path $python)) {
    Write-Host '==> Creating virtual environment (Python 3.10)' -ForegroundColor Cyan
    & py -3.10 -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw 'Failed to create the virtual environment.' }
}

Write-Host '==> Upgrading pip' -ForegroundColor Cyan
& $python -m pip install --upgrade pip --quiet

Write-Host '==> Installing mcmot (editable) with dev tooling' -ForegroundColor Cyan
& $python -m pip install -e ".[dev]"
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }

if ($WithStorage) {
    Write-Host '==> Installing the storage extra (SQLAlchemy, faiss-cpu)' -ForegroundColor Cyan
    & $python -m pip install -e ".[storage]"
    if ($LASTEXITCODE -ne 0) { throw 'storage extra installation failed.' }
}

if ($WithTorch) {
    Write-Host '==> Installing PyTorch 2.5.1 (CUDA 12.1) - 2.3 GB, be patient' -ForegroundColor Cyan
    # A long timeout and many retries: the wheel is large and the PyTorch index
    # drops slow connections. pip resumes a partial download, so re-running this
    # after a failure picks up where it stopped.
    & $python -m pip install `
        'torch==2.5.1' 'torchvision==0.20.1' `
        --index-url 'https://download.pytorch.org/whl/cu121' `
        --timeout 120 --retries 20
    if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed - re-run to resume.' }

    & $python -c "import torch; print('    torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"
}

if ($WithMl) {
    Write-Host '==> Installing the ml extra' -ForegroundColor Cyan
    & $python -m pip install -e ".[ml]"
    if ($LASTEXITCODE -ne 0) { throw 'ml extra installation failed.' }
}

if (-not (Test-Path (Join-Path $root '.env'))) {
    Copy-Item (Join-Path $root '.env.example') (Join-Path $root '.env')
    Write-Host '==> Created .env from .env.example' -ForegroundColor Green
}

Write-Host ''
Write-Host 'Environment ready.' -ForegroundColor Green
Write-Host '  Activate :  .\.venv\Scripts\Activate.ps1'
Write-Host '  Init DB  :  python scripts\init_db.py'
Write-Host '  Verify   :  python scripts\smoke_check.py'
Write-Host '  Test     :  pytest'
Write-Host '  Lint     :  ruff check . ; mypy'
Write-Host '  Serve    :  python -m mcmot.main --reload'
