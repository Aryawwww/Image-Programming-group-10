#Requires -Version 5.1
[CmdletBinding(DefaultParameterSetName = 'Module')]
param(
    [Parameter(ParameterSetName = 'Module', Mandatory = $true)]
    [string]$Module,

    [Parameter(ParameterSetName = 'Script', Mandatory = $true)]
    [string]$ScriptPath,

    [string[]]$Arguments = @(),

    [string]$ProjectRoot = ''
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = Split-Path -Parent $PSScriptRoot
}

$resolvedRoot = [System.IO.Path]::GetFullPath($ProjectRoot).TrimEnd('\', '/')
$rootPrefix = $resolvedRoot + [System.IO.Path]::DirectorySeparatorChar
$projectMarker = Join-Path $resolvedRoot 'pyproject.toml'
if (-not (Test-Path -LiteralPath $projectMarker -PathType Leaf)) {
    throw "Project root is invalid: $resolvedRoot"
}

$pythonExe = Join-Path $resolvedRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe -PathType Leaf)) {
    throw "Canonical virtual environment is missing. Run scripts\bootstrap.ps1 first."
}

$env:PYTHONHASHSEED = '42177'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$env:PLANTVILLAGE_PROJECT_ROOT = $resolvedRoot
$env:PYTHONPATH = Join-Path $resolvedRoot 'src'

Push-Location $resolvedRoot
try {
    if ($PSCmdlet.ParameterSetName -eq 'Module') {
        & $pythonExe -m $Module @Arguments
    }
    else {
        $resolvedScript = [System.IO.Path]::GetFullPath((Join-Path $resolvedRoot $ScriptPath))
        if (-not $resolvedScript.StartsWith($rootPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "ScriptPath must stay inside the project root."
        }
        if (-not (Test-Path -LiteralPath $resolvedScript -PathType Leaf)) {
            throw "Script does not exist: $resolvedScript"
        }
        & $pythonExe $resolvedScript @Arguments
    }
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
