$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    $Py = Join-Path $Root ".venv\bin\python"
}
Set-Location $Root
if (-not (Test-Path $Py)) { throw "No local venv. Run: python3 scripts/bootstrap.py" }
if ($args.Count -eq 0) {
  & $Py -m pvch2.cli check
} else {
  & $Py -m pvch2.cli check @args
}
exit $LASTEXITCODE
