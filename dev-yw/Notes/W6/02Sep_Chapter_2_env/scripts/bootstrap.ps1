# Recreate the local venv with whatever Python 3.13.x is on this machine.
# Do not hard-code E:\IT\Python\Python313 — Apple Silicon has Homebrew 3.13.x.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$bootstrap = Join-Path $Root "scripts\bootstrap.py"
$py = $null
foreach ($name in @("python3.13", "python3", "python")) {
    $cmd = Get-Command $name -ErrorAction SilentlyContinue
    if ($cmd) { $py = $cmd.Source; break }
}
if (-not $py) {
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($launcher) {
        & $launcher.Source -3.13 $bootstrap @args
        exit $LASTEXITCODE
    }
    throw "Python 3.13+ not found. Install it, or set PVCH2_PYTHON."
}
& $py $bootstrap @args
exit $LASTEXITCODE
