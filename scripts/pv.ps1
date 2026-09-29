#Requires -Version 5.1
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$CommandArgs = @($args)

if ($CommandArgs.Count -eq 0) {
    throw 'Provide a pv_pipeline command, for example: doctor --config configs/project_config.json'
}

& (Join-Path $PSScriptRoot 'run_deterministic.ps1') -Module pv_pipeline -Arguments $CommandArgs
exit $LASTEXITCODE
