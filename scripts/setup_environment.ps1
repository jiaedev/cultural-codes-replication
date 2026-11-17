param([Parameter(Mandatory=$true)][string]$PythonExecutable)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectRoot
try {
    & $PythonExecutable -c "import sys; assert sys.version_info[:2] == (3,11), sys.version"
    if ($LASTEXITCODE -ne 0) { throw 'Provide an existing Python 3.11 interpreter; tested with 3.11.17.' }
    if (-not (Test-Path -LiteralPath '.venv')) {
        & $PythonExecutable -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed' }
    }
    $venvPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
    & $venvPython -c "import sys; assert sys.prefix != sys.base_prefix and sys.version_info[:2] == (3,11)"
    if ($LASTEXITCODE -ne 0) { throw 'Existing .venv has an incompatible interpreter' }
    $env:PIP_CACHE_DIR = Join-Path $projectRoot '.cache\pip'
    & $venvPython -m pip install -r environment/requirements-win-cpu.lock.txt
    if ($LASTEXITCODE -ne 0) { throw 'Pinned installation failed' }
    & $venvPython -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed' }
} finally { Pop-Location }
