#requires -version 5.1
[CmdletBinding()]
param(
    [string]$InstallDir = "$env:ProgramData\GoodBadminton\venue-gateway",
    [string]$ConfigFile = "$env:ProgramData\GoodBadminton\venue-gateway.env"
)

$ErrorActionPreference = "Stop"
$python = Join-Path $InstallDir ".venv\Scripts\python.exe"
$agent = Join-Path $InstallDir "agent.py"
$logDir = Join-Path $InstallDir "logs"
$stdoutLog = Join-Path $logDir "venue-gateway.stdout.log"
$stderrLog = Join-Path $logDir "venue-gateway.stderr.log"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

foreach ($path in @($stdoutLog, $stderrLog)) {
    if ((Test-Path $path) -and (Get-Item $path).Length -gt 20MB) {
        Move-Item -Force $path "$path.1"
    }
}

if (-not (Test-Path $python) -or -not (Test-Path $agent) -or -not (Test-Path $ConfigFile)) {
    Add-Content -Encoding UTF8 $stderrLog "$(Get-Date -Format o) installation or configuration file missing"
    exit 2
}
if (Select-String -Quiet -SimpleMatch "SET_BY_" $ConfigFile) {
    Add-Content -Encoding UTF8 $stderrLog "$(Get-Date -Format o) configuration contains unresolved placeholders"
    exit 2
}

$env:GOOD_BADMINTON_VENUE_ENV_FILE = $ConfigFile
$env:PYTHONUNBUFFERED = "1"
Push-Location $InstallDir
$launcherErrorActionPreference = $ErrorActionPreference
try {
    # Windows PowerShell 5.1 can promote a native program's first stderr line
    # to a terminating NativeCommandError when ErrorActionPreference is Stop.
    # Let Python write its complete traceback to the redirected stderr log.
    $ErrorActionPreference = "Continue"
    & $python $agent 1>> $stdoutLog 2>> $stderrLog
    $exitCode = $LASTEXITCODE
} catch {
    Add-Content -Encoding UTF8 $stderrLog "$(Get-Date -Format o) launcher failure: $($_.Exception.Message)"
    $exitCode = 1
} finally {
    $ErrorActionPreference = $launcherErrorActionPreference
    Pop-Location
}
exit $exitCode
