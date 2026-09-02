#requires -version 5.1
[CmdletBinding()]
param(
    [string]$InstallDir = "$env:ProgramData\GoodBadminton\venue-gateway",
    [string]$ConfigFile = "$env:ProgramData\GoodBadminton\venue-gateway.env"
)

$ErrorActionPreference = "Stop"
$TaskName = "GoodBadmintonVenueGateway"
$TaskPath = "\GoodBadminton\"

function Assert-Administrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw "Run Windows PowerShell as Administrator, then execute this installer again."
    }
}

function Resolve-Python311 {
    $launcher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($launcher) {
        & $launcher.Source -3.11 -c "import struct, sys; assert sys.version_info[:2] == (3, 11) and struct.calcsize('P') * 8 == 64" 2>$null
        if ($LASTEXITCODE -eq 0) {
            return @{ Executable = $launcher.Source; PrefixArguments = @("-3.11") }
        }
    }
    $python = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($python) {
        & $python.Source -c "import struct, sys; assert sys.version_info[:2] == (3, 11) and struct.calcsize('P') * 8 == 64" 2>$null
        if ($LASTEXITCODE -eq 0) {
            return @{ Executable = $python.Source; PrefixArguments = @() }
        }
    }
    throw "64-bit Python 3.11 was not found. Install it from python.org with Python Launcher or Add Python to PATH enabled."
}

Assert-Administrator
$python = Resolve-Python311
$ffmpeg = Get-Command ffmpeg.exe -ErrorAction SilentlyContinue
if (-not $ffmpeg) {
    throw "ffmpeg.exe was not found. Install 64-bit FFmpeg with libx264 and add its bin directory to the system PATH."
}
$x264Encoder = & $ffmpeg.Source -hide_banner -encoders 2>&1 | Select-String -SimpleMatch "libx264"
if (-not $x264Encoder) {
    throw "The installed FFmpeg does not provide the libx264 encoder."
}

$sourceDir = $PSScriptRoot
$configDir = Split-Path -Parent $ConfigFile
$spoolDir = Join-Path $InstallDir "spool"
$logDir = Join-Path $InstallDir "logs"
New-Item -ItemType Directory -Force -Path $InstallDir, $configDir, $spoolDir, $logDir | Out-Null

$sourceFull = [IO.Path]::GetFullPath($sourceDir).TrimEnd('\')
$installFull = [IO.Path]::GetFullPath($InstallDir).TrimEnd('\')
if ($sourceFull -ne $installFull) {
    Copy-Item -Force (Join-Path $sourceDir "agent.py") $InstallDir
    foreach ($file in @("requirements.txt", "venue-gateway.env.example", "README.md", "VERSION")) {
        Copy-Item -Force (Join-Path $sourceDir $file) $InstallDir
    }
    foreach ($script in @("run-gateway.ps1", "start-gateway.ps1", "stop-gateway.ps1", "test-windows.ps1", "uninstall-windows.ps1")) {
        Copy-Item -Force (Join-Path $sourceDir $script) $InstallDir
    }
    $businessGatewayDir = Join-Path $InstallDir "business_gateway"
    New-Item -ItemType Directory -Force -Path $businessGatewayDir | Out-Null
    Copy-Item -Recurse -Force (Join-Path $sourceDir "business_gateway\*") $businessGatewayDir
}

$venvPython = Join-Path $InstallDir ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    & $python.Executable @($python.PrefixArguments) -m venv (Join-Path $InstallDir ".venv")
    if ($LASTEXITCODE -ne 0) { throw "Failed to create the Python virtual environment." }
}
& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Failed to upgrade pip." }
& $venvPython -m pip install -r (Join-Path $InstallDir "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "Failed to install Python dependencies." }

if (-not (Test-Path $ConfigFile)) {
    $config = Get-Content -Raw -Encoding UTF8 (Join-Path $InstallDir "venue-gateway.env.example")
    $config = $config.Replace("SPOOL_DIR=SET_BY_INSTALLER", "SPOOL_DIR=$spoolDir")
    $config = $config.Replace("FFMPEG_BIN=SET_BY_INSTALLER", "FFMPEG_BIN=$($ffmpeg.Source)")
    Set-Content -Encoding UTF8 -Path $ConfigFile -Value $config
    Write-Host "Created protected configuration template: $ConfigFile"
}

$currentUser = [Security.Principal.WindowsIdentity]::GetCurrent().Name
& icacls.exe $ConfigFile /inheritance:r /grant:r "*S-1-5-18:(F)" "*S-1-5-32-544:(F)" "$currentUser`:(F)" | Out-Null
if ($LASTEXITCODE -ne 0) { throw "Failed to restrict the configuration file ACL." }

$powerShellExe = Join-Path $PSHOME "powershell.exe"
$runScript = Join-Path $InstallDir "run-gateway.ps1"
$actionArgs = "-NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File `"$runScript`" -InstallDir `"$InstallDir`" -ConfigFile `"$ConfigFile`""
$action = New-ScheduledTaskAction -Execute $powerShellExe -Argument $actionArgs
$trigger = New-ScheduledTaskTrigger -AtStartup
$settings = New-ScheduledTaskSettingsSet -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit ([TimeSpan]::Zero) -StartWhenAvailable -MultipleInstances IgnoreNew
$principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest
Register-ScheduledTask -TaskName $TaskName -TaskPath $TaskPath -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Force | Out-Null

$hasPlaceholders = Select-String -Quiet -SimpleMatch "SET_BY_" $ConfigFile
if ($hasPlaceholders) {
    Disable-ScheduledTask -TaskName $TaskName -TaskPath $TaskPath | Out-Null
    Write-Host "Installation complete. The task remains disabled because configuration placeholders remain."
    Write-Host "Edit: $ConfigFile"
    Write-Host "Then run: & `"$InstallDir\test-windows.ps1`" -ProbeCamera"
    Write-Host "After validation run: & `"$InstallDir\start-gateway.ps1`""
} else {
    Enable-ScheduledTask -TaskName $TaskName -TaskPath $TaskPath | Out-Null
    Start-ScheduledTask -TaskName $TaskName -TaskPath $TaskPath
    Write-Host "Installation complete. Started $TaskName."
}
