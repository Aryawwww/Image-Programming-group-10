#Requires -Version 5.1
[CmdletBinding()]
param(
    [ValidateSet('Preflight', 'Runtime', 'Wheelhouse', 'Install', 'Verify', 'All')]
    [string]$Action = 'Preflight',

    [string]$ProjectRoot = '',

    [string]$PythonExecutable = '',

    [switch]$AllowOnline
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = Split-Path -Parent $PSScriptRoot
}

$script:ExpectedPython = '3.11.9'
$script:Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$script:ProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot).TrimEnd('\', '/')
$script:RootPrefix = $script:ProjectRoot + [System.IO.Path]::DirectorySeparatorChar
$script:RuntimeDir = Join-Path $script:ProjectRoot '.runtime\python311'
$script:RuntimePython = Join-Path $script:RuntimeDir 'python.exe'
$script:VenvDir = Join-Path $script:ProjectRoot '.venv'
$script:VenvPython = Join-Path $script:VenvDir 'Scripts\python.exe'
$script:CacheDir = Join-Path $script:ProjectRoot '.cache\bootstrap'
$script:InstallerPath = Join-Path $script:CacheDir 'python-3.11.9-amd64.exe'
$script:WheelhouseDir = Join-Path $script:ProjectRoot 'wheelhouse'
$script:ArtifactDir = Join-Path $script:ProjectRoot 'artifacts\environment'
$script:ManifestPath = Join-Path $script:ArtifactDir 'wheelhouse_manifest.json'
$script:ResolvedRequirements = Join-Path $script:ProjectRoot 'requirements\resolved.txt'
$script:DeploymentLockPath = Join-Path $script:ProjectRoot 'deployment.lock.json'

function Write-Stage {
    param([string]$Message)
    Write-Host "`n==> $Message" -ForegroundColor Cyan
}

function Assert-InProject {
    param([string]$Path)
    $fullPath = [System.IO.Path]::GetFullPath($Path)
    if (-not $fullPath.StartsWith($script:RootPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Generated path must stay inside the project root: $fullPath"
    }
    return $fullPath
}

function Ensure-Directory {
    param([string]$Path)
    $fullPath = Assert-InProject $Path
    if (-not (Test-Path -LiteralPath $fullPath -PathType Container)) {
        New-Item -ItemType Directory -Path $fullPath -Force | Out-Null
    }
}

function Write-Utf8Text {
    param(
        [string]$Path,
        [string]$Text
    )
    $fullPath = Assert-InProject $Path
    Ensure-Directory (Split-Path -Parent $fullPath)
    [System.IO.File]::WriteAllText($fullPath, $Text, $script:Utf8NoBom)
}

function Write-JsonFile {
    param(
        [string]$Path,
        [object]$Value
    )
    $json = $Value | ConvertTo-Json -Depth 30
    Write-Utf8Text -Path $Path -Text ($json + "`n")
}

function Invoke-Checked {
    param(
        [string]$Executable,
        [string[]]$Arguments,
        [string]$Description
    )
    Write-Host "-- $Description"
    & $Executable @Arguments
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "$Description failed with exit code $exitCode."
    }
}

function Assert-ExactPython {
    param([string]$Executable)
    if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) {
        throw "Python executable does not exist: $Executable"
    }
    $resolvedExecutable = (Resolve-Path -LiteralPath $Executable).Path
    $versionOutput = & $resolvedExecutable -c 'import platform; print(platform.python_version())'
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to inspect Python: $resolvedExecutable"
    }
    $actualVersion = ([string]($versionOutput | Select-Object -Last 1)).Trim()
    if ($actualVersion -ne $script:ExpectedPython) {
        throw "Python $script:ExpectedPython is required; found $actualVersion at $resolvedExecutable."
    }
    return $resolvedExecutable
}

function Get-DeploymentLock {
    if (-not (Test-Path -LiteralPath $script:DeploymentLockPath -PathType Leaf)) {
        throw "Missing deployment lock: $script:DeploymentLockPath"
    }
    return Get-Content -LiteralPath $script:DeploymentLockPath -Raw | ConvertFrom-Json
}

function Get-RequirementRows {
    param([string]$Path)
    return @(
        Get-Content -LiteralPath $Path | ForEach-Object { $_.Trim() } |
            Where-Object { $_ -and -not $_.StartsWith('#') }
    )
}

function Invoke-Preflight {
    Write-Stage 'Preflight: validate immutable deployment inputs'
    Ensure-Directory $script:ArtifactDir

    $checks = New-Object 'System.Collections.Generic.List[object]'
    $errors = New-Object 'System.Collections.Generic.List[string]'
    function Add-Check {
        param(
            [string]$Name,
            [bool]$Passed,
            [object]$Detail
        )
        $checks.Add([pscustomobject]@{
                name = $Name
                passed = $Passed
                detail = $Detail
            }) | Out-Null
        if (-not $Passed) {
            $errors.Add("${Name}: $Detail") | Out-Null
        }
    }

    $requiredFiles = @(
        '.python-version',
        'pyproject.toml',
        'deployment.lock.json',
        'configs/project_config.json',
        'configs/classes.json',
        'requirements/bootstrap.txt',
        'requirements/reference.txt',
        'requirements/resolved.txt',
        'scripts/bootstrap.ps1',
        'scripts/run_deterministic.ps1',
        'scripts/verify_environment.py',
        'scripts/verify_wheelhouse.py',
        'scripts/write_wheelhouse_manifest.py'
    )
    $missing = @(
        $requiredFiles | Where-Object {
            -not (Test-Path -LiteralPath (Join-Path $script:ProjectRoot $_) -PathType Leaf)
        }
    )
    Add-Check 'required_files' ($missing.Count -eq 0) ([pscustomobject]@{ missing = $missing })

    if ($missing.Count -eq 0) {
        $lock = Get-DeploymentLock
        Add-Check 'lock_schema' ($lock.schema_version -eq '1.0.0') $lock.schema_version
        Add-Check 'locked_python' ($lock.python.version -eq $script:ExpectedPython) $lock.python.version
        Add-Check 'locked_architecture' ($lock.python.architecture -eq 'amd64') $lock.python.architecture

        $hashMismatches = New-Object 'System.Collections.Generic.List[object]'
        foreach ($property in @($lock.locked_files.PSObject.Properties)) {
            $target = Join-Path $script:ProjectRoot $property.Name
            $actualHash = $null
            if (Test-Path -LiteralPath $target -PathType Leaf) {
                $actualHash = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant()
            }
            if ($actualHash -ne ([string]$property.Value).ToLowerInvariant()) {
                $hashMismatches.Add([pscustomobject]@{
                        path = $property.Name
                        expected = $property.Value
                        actual = $actualHash
                    }) | Out-Null
            }
        }
        Add-Check 'locked_file_hashes' ($hashMismatches.Count -eq 0) ([pscustomobject]@{ mismatches = $hashMismatches })

        $pythonVersionFile = (Get-Content -LiteralPath (Join-Path $script:ProjectRoot '.python-version') -Raw).Trim()
        Add-Check 'python_version_file' ($pythonVersionFile -eq $script:ExpectedPython) $pythonVersionFile

        $config = Get-Content -LiteralPath (Join-Path $script:ProjectRoot 'configs\project_config.json') -Raw | ConvertFrom-Json
        $classes = Get-Content -LiteralPath (Join-Path $script:ProjectRoot 'configs\classes.json') -Raw | ConvertFrom-Json
        $classIds = @($classes.classes | ForEach-Object { [int]$_.class_id })
        Add-Check 'seed' ([int]$config.seed -eq 42177) $config.seed
        Add-Check 'class_ids' (($classIds -join ',') -eq '0,1,2,3') $classIds
        $counts = $config.dataset.selected_count_per_class
        $splitCountsPass = (
            [int]$counts.train -eq 630 -and
            [int]$counts.val -eq 135 -and
            [int]$counts.test -eq 135
        )
        Add-Check 'split_counts' $splitCountsPass $counts

        $referenceRows = Get-RequirementRows (Join-Path $script:ProjectRoot 'requirements\reference.txt')
        $bootstrapRows = Get-RequirementRows (Join-Path $script:ProjectRoot 'requirements\bootstrap.txt')
        $allRowsPinned = (@($referenceRows + $bootstrapRows | Where-Object { $_ -notmatch '^[A-Za-z0-9_.-]+==[^\s]+$' }).Count -eq 0)
        Add-Check 'direct_requirement_counts' ($referenceRows.Count -eq 15 -and $bootstrapRows.Count -eq 3) ([pscustomobject]@{
                reference = $referenceRows.Count
                bootstrap = $bootstrapRows.Count
            })
        Add-Check 'direct_requirements_exactly_pinned' $allRowsPinned $allRowsPinned
    }

    $report = [ordered]@{
        schema_version = '1.0.0'
        generated_at_utc = [DateTime]::UtcNow.ToString('o')
        action = 'Preflight'
        status = $(if ($errors.Count -eq 0) { 'pass' } else { 'fail' })
        checks = $checks
        errors = $errors
    }
    $reportPath = Join-Path $script:ArtifactDir 'preflight_report.json'
    Write-JsonFile -Path $reportPath -Value $report
    if ($errors.Count -ne 0) {
        throw "Preflight failed. See $reportPath"
    }
    Write-Host "Preflight passed: $reportPath" -ForegroundColor Green
}

function Test-Installer {
    param([string]$Path)
    $lock = Get-DeploymentLock
    $signature = Get-AuthenticodeSignature -LiteralPath $Path
    $subject = if ($null -ne $signature.SignerCertificate) { $signature.SignerCertificate.Subject } else { '' }
    if ($signature.Status.ToString() -ne 'Valid') {
        throw "Python installer Authenticode status is $($signature.Status): $Path"
    }
    if ($subject -notlike "*$($lock.python.signer_subject_contains)*") {
        throw "Unexpected Python installer signer: $subject"
    }
    $actualHash = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    $expectedHash = $lock.python.installer_sha256
    if ($null -ne $expectedHash -and -not [string]::IsNullOrWhiteSpace([string]$expectedHash)) {
        if ($actualHash -ne ([string]$expectedHash).ToLowerInvariant()) {
            throw "Python installer SHA-256 mismatch. Expected $expectedHash; found $actualHash."
        }
    }
    return [pscustomobject]@{
        sha256 = $actualHash
        bytes = (Get-Item -LiteralPath $Path).Length
        signature_status = $signature.Status.ToString()
        signer_subject = $subject
        signer_thumbprint = $signature.SignerCertificate.Thumbprint
    }
}

function Ensure-PythonRuntime {
    Write-Stage 'Runtime: provision exact Python 3.11.9'
    if (-not [string]::IsNullOrWhiteSpace($PythonExecutable)) {
        return Assert-ExactPython ([System.IO.Path]::GetFullPath($PythonExecutable))
    }
    if (Test-Path -LiteralPath $script:RuntimePython -PathType Leaf) {
        return Assert-ExactPython $script:RuntimePython
    }

    $lock = Get-DeploymentLock
    Ensure-Directory $script:CacheDir
    if (-not (Test-Path -LiteralPath $script:InstallerPath -PathType Leaf)) {
        if (-not $AllowOnline) {
            throw "Python 3.11.9 is absent. Re-run Runtime or All with -AllowOnline."
        }
        $downloadPath = Join-Path $script:CacheDir ("python-3.11.9-download-{0}.exe" -f [Guid]::NewGuid().ToString('N'))
        Write-Host "-- Download signed installer from $($lock.python.installer_url)"
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -Uri $lock.python.installer_url -OutFile $downloadPath -UseBasicParsing
        $null = Test-Installer $downloadPath
        Move-Item -LiteralPath $downloadPath -Destination $script:InstallerPath
    }

    $installer = Test-Installer $script:InstallerPath
    $installerRecord = [ordered]@{
        schema_version = '1.0.0'
        verified_at_utc = [DateTime]::UtcNow.ToString('o')
        source_url = $lock.python.installer_url
        file = (Split-Path -Leaf $script:InstallerPath)
        bytes = $installer.bytes
        sha256 = $installer.sha256
        authenticode = [ordered]@{
            status = $installer.signature_status
            signer_subject = $installer.signer_subject
            signer_thumbprint = $installer.signer_thumbprint
        }
    }
    Write-JsonFile -Path (Join-Path $script:ArtifactDir 'python_installer_lock.json') -Value $installerRecord

    Ensure-Directory $script:RuntimeDir
    $installerArguments = @(
        '/quiet',
        'InstallAllUsers=0',
        ("TargetDir=`"{0}`"" -f $script:RuntimeDir),
        'Include_launcher=0',
        'Include_test=0',
        'Include_doc=0',
        'Include_tcltk=0',
        'Include_pip=1',
        'PrependPath=0',
        'AppendPath=0',
        'AssociateFiles=0',
        'Shortcuts=0'
    )
    Write-Host "-- Install Python into $script:RuntimeDir"
    $process = Start-Process -FilePath $script:InstallerPath -ArgumentList $installerArguments -Wait -PassThru -WindowStyle Hidden
    if ($process.ExitCode -notin @(0, 3010)) {
        throw "Python installer failed with exit code $($process.ExitCode)."
    }
    return Assert-ExactPython $script:RuntimePython
}

function Ensure-VirtualEnvironment {
    if (Test-Path -LiteralPath $script:VenvPython -PathType Leaf) {
        return Assert-ExactPython $script:VenvPython
    }
    if (Test-Path -LiteralPath $script:VenvDir) {
        throw "An incomplete virtual environment exists at $script:VenvDir. It was preserved; move it aside before retrying."
    }
    $basePython = Ensure-PythonRuntime
    Write-Stage 'Runtime: create project-local virtual environment'
    Invoke-Checked -Executable $basePython -Arguments @('-m', 'venv', $script:VenvDir) -Description 'Create .venv'
    return Assert-ExactPython $script:VenvPython
}

function Invoke-WheelhouseVerification {
    param([string]$Python)
    Invoke-Checked -Executable $Python -Arguments @(
        (Join-Path $script:ProjectRoot 'scripts\verify_wheelhouse.py'),
        '--wheelhouse', $script:WheelhouseDir,
        '--manifest', $script:ManifestPath,
        '--requirements', $script:ResolvedRequirements,
        '--deployment-lock', $script:DeploymentLockPath
    ) -Description 'Verify wheelhouse file set and SHA-256 values'
}

function Ensure-Wheelhouse {
    $venvPython = Ensure-VirtualEnvironment
    Write-Stage 'Wheelhouse: build or verify the complete offline dependency set'

    if (Test-Path -LiteralPath $script:WheelhouseDir -PathType Container) {
        if (-not (Test-Path -LiteralPath $script:ManifestPath -PathType Leaf) -or
            -not (Test-Path -LiteralPath $script:ResolvedRequirements -PathType Leaf)) {
            throw "Wheelhouse exists without its manifest or resolved lock. It was preserved; move it aside before retrying."
        }
        Invoke-WheelhouseVerification $venvPython
        Write-Host 'Existing wheelhouse is valid; download skipped.' -ForegroundColor Green
        return
    }
    if (-not $AllowOnline) {
        throw "Wheelhouse is absent. Re-run Wheelhouse or All with -AllowOnline."
    }

    $lock = Get-DeploymentLock
    $buildRoot = Join-Path $script:ProjectRoot ('.cache\wheelhouse-build\' + [Guid]::NewGuid().ToString('N'))
    $stageWheelhouse = Join-Path $buildRoot 'wheelhouse'
    $stageManifest = Join-Path $buildRoot 'wheelhouse_manifest.json'
    $stageRequirements = Join-Path $buildRoot 'resolved.txt'
    Ensure-Directory $stageWheelhouse

    $downloadArguments = @(
        '-m', 'pip', 'download',
        '--disable-pip-version-check',
        '--no-cache-dir',
        '--only-binary=:all:',
        '--index-url', $lock.python.package_index_url,
        '--dest', $stageWheelhouse
    )
    if (Test-Path -LiteralPath $script:ResolvedRequirements -PathType Leaf) {
        $downloadArguments += @('--require-hashes', '-r', $script:ResolvedRequirements)
    }
    else {
        $downloadArguments += @(
            '-r', (Join-Path $script:ProjectRoot 'requirements\bootstrap.txt'),
            '-r', (Join-Path $script:ProjectRoot 'requirements\reference.txt')
        )
    }
    Invoke-Checked -Executable $venvPython -Arguments $downloadArguments -Description 'Download canonical wheels'
    Invoke-Checked -Executable $venvPython -Arguments @(
        (Join-Path $script:ProjectRoot 'scripts\write_wheelhouse_manifest.py'),
        '--wheelhouse', $stageWheelhouse,
        '--output', $stageManifest,
        '--requirements-output', $stageRequirements
    ) -Description 'Generate wheelhouse manifest and hashed transitive lock'

    if (Test-Path -LiteralPath $script:ResolvedRequirements -PathType Leaf) {
        $existingHash = (Get-FileHash -LiteralPath $script:ResolvedRequirements -Algorithm SHA256).Hash
        $generatedHash = (Get-FileHash -LiteralPath $stageRequirements -Algorithm SHA256).Hash
        if ($existingHash -ne $generatedHash) {
            throw "Rebuilt wheelhouse produced a different resolved dependency lock. Staging files were preserved at $buildRoot."
        }
    }
    else {
        Copy-Item -LiteralPath $stageRequirements -Destination $script:ResolvedRequirements
    }
    Copy-Item -LiteralPath $stageManifest -Destination $script:ManifestPath -Force
    Move-Item -LiteralPath $stageWheelhouse -Destination $script:WheelhouseDir
    Invoke-WheelhouseVerification $venvPython
}

function Invoke-EnvironmentInstall {
    $venvPython = Ensure-VirtualEnvironment
    Ensure-Wheelhouse
    Write-Stage 'Install: install exclusively from the verified wheelhouse'
    Invoke-Checked -Executable $venvPython -Arguments @(
        '-m', 'pip', 'install',
        '--disable-pip-version-check',
        '--no-index',
        '--find-links', $script:WheelhouseDir,
        '--require-hashes',
        '-r', $script:ResolvedRequirements
    ) -Description 'Install fully locked environment'
    Invoke-Checked -Executable $venvPython -Arguments @(
        '-m', 'pip', 'install',
        '--disable-pip-version-check',
        '--no-index',
        '--no-deps',
        '--no-build-isolation',
        '--editable', $script:ProjectRoot
    ) -Description 'Install the local project as an offline editable package'

    $pipCheckOutput = @(& $venvPython -m pip check 2>&1 | ForEach-Object { [string]$_ })
    $pipCheckExitCode = $LASTEXITCODE
    Write-Utf8Text -Path (Join-Path $script:ArtifactDir 'pip_check.txt') -Text (($pipCheckOutput -join "`n") + "`n")
    if ($pipCheckExitCode -ne 0) {
        throw "pip check failed. See artifacts\environment\pip_check.txt"
    }

    $freezeOutput = @(& $venvPython -m pip freeze --all 2>&1 | ForEach-Object { [string]$_ })
    $freezeExitCode = $LASTEXITCODE
    Write-Utf8Text -Path (Join-Path $script:ArtifactDir 'pip_freeze.txt') -Text (($freezeOutput -join "`n") + "`n")
    if ($freezeExitCode -ne 0) {
        throw 'pip freeze failed.'
    }
    Copy-Item -LiteralPath $script:ResolvedRequirements -Destination (Join-Path $script:ArtifactDir 'requirements.resolved.txt') -Force
}

function Invoke-EnvironmentVerification {
    $venvPython = Assert-ExactPython $script:VenvPython
    Write-Stage 'Verify: deterministic imports, inventory, and PyTorch smoke test'
    Invoke-WheelhouseVerification $venvPython
    $env:PYTHONHASHSEED = '42177'
    $env:OMP_NUM_THREADS = '1'
    $env:MKL_NUM_THREADS = '1'
    $env:OPENBLAS_NUM_THREADS = '1'
    $env:NUMEXPR_NUM_THREADS = '1'
    $env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
    $env:PLANTVILLAGE_PROJECT_ROOT = $script:ProjectRoot
    Invoke-Checked -Executable $venvPython -Arguments @(
        (Join-Path $script:ProjectRoot 'scripts\verify_environment.py'),
        '--project-root', $script:ProjectRoot,
        '--mode', 'strict',
        '--output', (Join-Path $script:ArtifactDir 'environment_report.json')
    ) -Description 'Run strict environment verification'
}

if (-not (Test-Path -LiteralPath (Join-Path $script:ProjectRoot 'pyproject.toml') -PathType Leaf)) {
    throw "Invalid project root: $script:ProjectRoot"
}

Invoke-Preflight
switch ($Action) {
    'Preflight' {
        break
    }
    'Runtime' {
        $null = Ensure-VirtualEnvironment
        break
    }
    'Wheelhouse' {
        Ensure-Wheelhouse
        break
    }
    'Install' {
        Invoke-EnvironmentInstall
        break
    }
    'Verify' {
        Invoke-EnvironmentVerification
        break
    }
    'All' {
        $null = Ensure-VirtualEnvironment
        Ensure-Wheelhouse
        Invoke-EnvironmentInstall
        Invoke-EnvironmentVerification
        break
    }
}

Write-Host "`nDeployment action '$Action' completed successfully." -ForegroundColor Green
