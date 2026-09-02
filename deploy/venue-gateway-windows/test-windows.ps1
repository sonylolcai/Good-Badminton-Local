#requires -version 5.1
[CmdletBinding()]
param(
    [string]$InstallDir = "$env:ProgramData\GoodBadminton\venue-gateway",
    [string]$ConfigFile = "$env:ProgramData\GoodBadminton\venue-gateway.env",
    [switch]$ProbeCamera
)

$ErrorActionPreference = "Stop"
$failed = $false
function Pass([string]$message) { Write-Host "[PASS] $message" -ForegroundColor Green }
function Fail([string]$message) { Write-Host "[FAIL] $message" -ForegroundColor Red; $script:failed = $true }

if ([Environment]::Is64BitOperatingSystem) { Pass "64-bit Windows" } else { Fail "64-bit Windows is required" }
$python = Join-Path $InstallDir ".venv\Scripts\python.exe"
if (Test-Path $python) {
    $version = & $python -c "import platform, struct, sys; assert sys.version_info[:2] == (3, 11) and struct.calcsize('P') * 8 == 64; print(platform.python_version())"
    if ($LASTEXITCODE -eq 0) { Pass "Python $version 64-bit virtual environment" } else { Fail "A working 64-bit Python 3.11 virtual environment is required" }
} else { Fail "Installed Python virtual environment was not found" }

if (-not (Test-Path $ConfigFile)) {
    Fail "Configuration file does not exist: $ConfigFile"
    exit 1
}
if (Select-String -Quiet -SimpleMatch "SET_BY_" $ConfigFile) { Fail "Configuration contains unresolved SET_BY_ placeholders" } else { Pass "Configuration placeholders are resolved" }

$config = @{}
foreach ($raw in Get-Content -Encoding UTF8 $ConfigFile) {
    $line = $raw.Trim()
    if ($line -and -not $line.StartsWith("#") -and $line.Contains("=")) {
        $parts = $line -split '=', 2
        $config[$parts[0].Trim()] = $parts[1].Trim()
    }
}

foreach ($key in @("EDGE_GATEWAY_URL", "EDGE_DEVICE_ID", "EDGE_CAMERA_ID", "EDGE_DEVICE_SECRET", "CAMERA_RTSP_URL", "COURT_CORNERS_JSON", "SPOOL_DIR", "FFMPEG_BIN")) {
    if (-not $config.ContainsKey($key) -or -not $config[$key]) { Fail "Missing configuration key $key" }
}

if ($config.ContainsKey("SPOOL_DIR")) {
    try {
        New-Item -ItemType Directory -Force -Path $config["SPOOL_DIR"] | Out-Null
        $drive = Get-Item $config["SPOOL_DIR"]
        $freeGB = [math]::Round((Get-PSDrive $drive.PSDrive.Name).Free / 1GB, 1)
        if ($freeGB -ge 5) { Pass "Spool directory is writable with ${freeGB}GB free" } else { Fail "Less than 5GB disk space remains" }
    } catch { Fail "Spool directory is not writable" }
}

if ($config.ContainsKey("FFMPEG_BIN") -and (Test-Path $config["FFMPEG_BIN"])) {
    $ffmpegBin = $config["FFMPEG_BIN"]
    $x264Encoder = & $ffmpegBin -hide_banner -encoders 2>&1 | Select-String -SimpleMatch "libx264"
    if ($LASTEXITCODE -eq 0 -and $x264Encoder) { Pass "FFmpeg executable and libx264 encoder" } else { Fail "FFmpeg does not provide the libx264 encoder" }
} else { Fail "FFMPEG_BIN does not exist" }

if ($config.ContainsKey("EDGE_GATEWAY_URL")) {
    try {
        $uri = [Uri]$config["EDGE_GATEWAY_URL"]
        $port = if ($uri.Port -gt 0) { $uri.Port } else { 443 }
        if (Test-NetConnection -ComputerName $uri.Host -Port $port -InformationLevel Quiet) {
            Pass "Business server $($uri.Host):$port is reachable"
        } else { Fail "Business server $($uri.Host):$port is unreachable" }
    } catch { Fail "EDGE_GATEWAY_URL is invalid" }
}

if ($ProbeCamera -and $config.ContainsKey("CAMERA_RTSP_URL") -and $config.ContainsKey("FFMPEG_BIN")) {
    $ffmpegBin = $config["FFMPEG_BIN"]
    $ffprobe = Join-Path (Split-Path -Parent $ffmpegBin) "ffprobe.exe"
    if (-not (Test-Path $ffprobe)) {
        Fail "ffprobe.exe was not found; camera probe cannot run"
    } else {
        & $ffprobe -v error -rw_timeout 10000000 -rtsp_transport tcp -select_streams v:0 -show_entries stream=codec_name,width,height,r_frame_rate -of json $config["CAMERA_RTSP_URL"]
        if ($LASTEXITCODE -eq 0) { Pass "Camera RTSP video stream is readable" } else { Fail "Camera RTSP video stream is not readable" }
    }
}

try {
    $timeStatus = & w32tm.exe /query /status 2>&1
    if ($LASTEXITCODE -eq 0) { Pass "Windows Time service is available" } else { Fail "Windows Time service is unhealthy; signatures allow only 120 seconds skew" }
} catch { Fail "Could not check Windows time synchronization" }

if ($failed) { exit 1 }
Write-Host "All checks passed." -ForegroundColor Green
