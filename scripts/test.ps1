#Requires -Version 5.1
[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

& (Join-Path $PSScriptRoot 'run_deterministic.ps1') -Module pytest -Arguments @('-q')
exit $LASTEXITCODE
