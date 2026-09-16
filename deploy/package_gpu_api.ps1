[CmdletBinding()]
param(
    # The file to upload through the GPU provider's browser upload page.
    [string]$OutputPath = ''
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$packageName = 'good-badminton-gpu-api'
if ([string]::IsNullOrWhiteSpace($OutputPath)) {
    $OutputPath = Join-Path $PSScriptRoot "$packageName-upload.zip"
}
$stagingRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("good-badminton-gpu-package-" + [guid]::NewGuid().ToString('N'))
$packageRoot = Join-Path $stagingRoot $packageName

# The shared API owns both allow-listed sport profiles. Package only its GPU
# runtime, not WebUI, business services, evaluation tools, data, or secrets.
$sourceRoots = @('api', 'badminton_analysis', 'good_badminton_contracts')
$requiredFiles = @(
    '.gpu-api.env.example',
    'requirements.txt',
    'simhei.ttf',
    'deploy/install_lap.sh',
    'deploy/refresh_gpu_api_from_zip.sh',
    'deploy/run_tracknet_v3_ab.sh',
    'deploy/setup_tracknet_v3_ab.sh',
    'deploy/start_gpu_api_container.sh',
    'deploy/stop_gpu_api_container.sh'
)

function Copy-SourceFile {
    param([Parameter(Mandatory = $true)][string]$RelativePath)

    $source = Join-Path $repoRoot $RelativePath
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "Required source file not found: $RelativePath"
    }
    $destination = Join-Path $packageRoot $RelativePath
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $destination) | Out-Null

    # Shell files need Linux line endings; source files are left untouched.
    if ($RelativePath.EndsWith('.sh', [System.StringComparison]::OrdinalIgnoreCase)) {
        $unixText = [System.IO.File]::ReadAllText($source).Replace("`r`n", "`n").Replace("`r", "`n")
        [System.IO.File]::WriteAllText($destination, $unixText, [System.Text.UTF8Encoding]::new($false))
    }
    else {
        Copy-Item -LiteralPath $source -Destination $destination -Force
    }
}

function Test-DeployablePath {
    param([Parameter(Mandatory = $true)][string]$RelativePath)

    $normalized = $RelativePath.Replace('\\', '/')
    if ($normalized -match '(^|/)(\.venv|venv|weights|api_data|outputs|videos|__pycache__)(/|$)') {
        return $false
    }
    return -not ($normalized -match '(^|/)\.env($|\.)')
}

try {
    New-Item -ItemType Directory -Force -Path $packageRoot | Out-Null

    foreach ($sourceRoot in $sourceRoots) {
        $trackedFiles = & git -C $repoRoot ls-files -- $sourceRoot
        if ($LASTEXITCODE -ne 0) {
            throw 'git ls-files failed; run this script from a Git checkout.'
        }
        foreach ($relativePath in $trackedFiles) {
            if (Test-DeployablePath -RelativePath $relativePath) {
                Copy-SourceFile -RelativePath $relativePath
            }
        }
    }

    foreach ($relativePath in $requiredFiles) {
        Copy-SourceFile -RelativePath $relativePath
    }

    # Include new, uncommitted runtime modules while keeping the same allow-list.
    foreach ($sourceRoot in $sourceRoots) {
        $untrackedFiles = & git -C $repoRoot ls-files --others --exclude-standard -- $sourceRoot
        if ($LASTEXITCODE -ne 0) {
            throw 'git ls-files --others failed; cannot safely include new runtime source.'
        }
        foreach ($relativePath in $untrackedFiles) {
            if ((Test-DeployablePath -RelativePath $relativePath) -and $relativePath -match '\.(py|json|ya?ml)$') {
                Copy-SourceFile -RelativePath $relativePath
            }
        }
    }

    foreach ($relativePath in @(
        'api/app.py',
        'api/vision_profiles.py',
        'deploy/refresh_gpu_api_from_zip.sh',
        'deploy/start_gpu_api_container.sh',
        'deploy/stop_gpu_api_container.sh'
    )) {
        if (-not (Test-Path -LiteralPath (Join-Path $packageRoot $relativePath) -PathType Leaf)) {
            throw "Package validation failed: missing $relativePath"
        }
    }

    $absoluteOutputPath = [System.IO.Path]::GetFullPath($OutputPath)
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $absoluteOutputPath) | Out-Null
    if (Test-Path -LiteralPath $absoluteOutputPath) {
        Remove-Item -LiteralPath $absoluteOutputPath -Force
    }

    Add-Type -AssemblyName System.IO.Compression
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zip = [System.IO.Compression.ZipFile]::Open($absoluteOutputPath, [System.IO.Compression.ZipArchiveMode]::Create)
    try {
        Get-ChildItem -LiteralPath $packageRoot -File -Recurse | ForEach-Object {
            $relativePath = $_.FullName.Substring($packageRoot.Length).TrimStart('\', '/').Replace('\', '/')
            [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
                $zip,
                $_.FullName,
                "$packageName/$relativePath",
                [System.IO.Compression.CompressionLevel]::Optimal
            ) | Out-Null
        }
    }
    finally {
        $zip.Dispose()
    }

    $item = Get-Item -LiteralPath $absoluteOutputPath
    Write-Host "Created GPU deployment package: $($item.FullName)"
    Write-Host "Size: $([math]::Round($item.Length / 1MB, 2)) MiB"
    Write-Host 'Upload it to /root/good-badminton-gpu-api-upload.zip, then run:'
    Write-Host 'bash /root/good-badminton-gpu-api/deploy/refresh_gpu_api_from_zip.sh'
    Write-Host 'First deployment only:'
    Write-Host 'unzip -p /root/good-badminton-gpu-api-upload.zip good-badminton-gpu-api/deploy/refresh_gpu_api_from_zip.sh | bash -s --'
}
finally {
    if (Test-Path -LiteralPath $stagingRoot) {
        Remove-Item -LiteralPath $stagingRoot -Recurse -Force
    }
}
