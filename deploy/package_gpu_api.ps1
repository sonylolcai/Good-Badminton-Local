[CmdletBinding()]
param(
    # The file to upload through the GPU provider's browser upload page.
    [string]$OutputPath = '',

    # Produce an independently deployable, fixed-sport package. The source
    # core is shared, but the archive name and server-side refresh target are
    # never interchangeable.
    [ValidateSet('badminton', 'tennis')]
    [string]$Sport = 'badminton',

    # Adds extended annotation and benchmark tooling.  The primary TrackNet
    # runtime adapters are always included below; no upstream source or model
    # weights are ever placed in this package.
    [switch]$IncludeTrackNetABTools,

    # Include the repository's current badminton YOLO-ball checkpoint only for
    # an explicitly labelled tennis trial. The ball model is never
    # presented as trained tennis evidence. This one checkpoint is copied to
    # persistent server weights on refresh; normal source packages exclude weights.
    [switch]$IncludeExperimentalTennisBallModel
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if ([string]::IsNullOrWhiteSpace($OutputPath)) {
    # Parameter defaults are evaluated before PowerShell reliably populates
    # $PSScriptRoot for this script. Resolve the conventional output path only
    # after entering the script body so the documented no-argument command
    # works in both Windows PowerShell 5.1 and PowerShell 7.
    $OutputPath = Join-Path $PSScriptRoot ("good-$Sport-gpu-api-upload.zip")
}
$stagingRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("good-badminton-gpu-package-" + [guid]::NewGuid().ToString('N'))
$packageName = "good-$Sport-gpu-api"
$packageRoot = Join-Path $stagingRoot $packageName

# The GPU instance has no reliable public egress.  This archive is therefore
# built from the *current working tree*, including tracked local modifications,
# rather than from `git archive HEAD`.  It intentionally excludes secrets,
# model weights, virtual environments, job artifacts and unrelated untracked
# experiments.  We only need source code and deployment files for an upgrade.
$requiredExtraFiles = @(
    'deploy/refresh_gpu_api_from_zip.sh',
    # Fixed-sport pure GPU launchers must be present even when this archive is
    # built before the current working tree has been committed.
    'deploy/start_sport_gpu_container.sh',
    'deploy/start_badminton_gpu_container.sh',
    'deploy/start_tennis_gpu_container.sh',
    'deploy/install_lap.sh',
    'deploy/package_gpu_api.ps1',
    'deploy/run_performance_gate.sh',
    'deploy/setup_tracknet_v3_ab.sh',
    'deploy/run_tracknet_v3_ab.sh',
    # TrackNet is the selectable primary shuttle source on the current branch.
    # These adapters are application code, not external model source or weights.
    'badminton_analysis/detection/tracknet_v3.py',
    'badminton_analysis/analysis/huji_play_state.py',
    # The package is built from a working tree while several runtime modules
    # may be newly created before their review commit. Keep direct API imports
    # explicit so deployment never omits a required local module merely because
    # `git ls-files` does not list untracked files.
    'badminton_analysis/cancellation.py',
    'badminton_analysis/visualization/spatial_player_positions.py',
    # The asynchronous GPU job worker imports this pipeline directly. Keep it
    # explicit so a healthy API cannot be packaged without its analysis entry.
    'webui/pipeline.py',
    'evaluation/shuttle_tracknet_ab/fast_predict_tracknet_v3.py',
    'evaluation/shuttle_tracknet_ab/run_tracknet_v3.py',
    # The performance gate is intentionally dependency-free and is run after
    # a relevant GPU deployment against the completed job trace.
    'evaluation/performance/__init__.py',
    'evaluation/performance/performance_gate.py',
    'evaluation/performance/rtx_3090_production_v1.json',
    'evaluation/performance/README.md',
    # Local continuity-test instructions are intentionally shipped with the
    # source archive so the GPU and business operators use one session/order
    # contract when validating a new deployment.
    'docs/STREAM_CONTINUITY_TEST.md'
)

$trackNetABFiles = @(
    'evaluation/shuttle_tracknet_ab/__init__.py',
    'evaluation/shuttle_tracknet_ab/annotations.py',
    'evaluation/shuttle_tracknet_ab/fast_predict_tracknet_v3.py',
    'evaluation/shuttle_tracknet_ab/init_annotation_set.py',
    'evaluation/shuttle_tracknet_ab/metrics.py',
    'evaluation/shuttle_tracknet_ab/prediction_io.py',
    'evaluation/shuttle_tracknet_ab/run_ab_benchmark.py',
    'evaluation/shuttle_tracknet_ab/run_tracknet_v3.py'
)

function Copy-SourceFile {
    param([Parameter(Mandatory = $true)][string]$RelativePath)

    $source = Join-Path $repoRoot $RelativePath
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "Required source file not found: $RelativePath"
    }
    $destination = Join-Path $packageRoot $RelativePath
    $destinationDir = Split-Path -Parent $destination
    New-Item -ItemType Directory -Force -Path $destinationDir | Out-Null

    # Shell files authored on Windows must use LF in the Linux package.  The
    # source working tree is deliberately left untouched; only the staged copy
    # is normalised so the server never fails with `/usr/bin/env: bash\r`.
    if ($RelativePath.EndsWith('.sh', [System.StringComparison]::OrdinalIgnoreCase)) {
        $unixText = [System.IO.File]::ReadAllText($source).Replace("`r`n", "`n").Replace("`r", "`n")
        [System.IO.File]::WriteAllText($destination, $unixText, [System.Text.UTF8Encoding]::new($false))
    }
    else {
        Copy-Item -LiteralPath $source -Destination $destination -Force
    }
}

try {
    New-Item -ItemType Directory -Force -Path $packageRoot | Out-Null

    $trackedFiles = & git -C $repoRoot ls-files
    if ($LASTEXITCODE -ne 0) {
        throw 'git ls-files failed; run this script from a Git checkout.'
    }

    foreach ($relativePath in $trackedFiles) {
        # Deployment never transports persisted or generated data.  The server
        # owns these paths under /root/good-badminton-gpu-api-state instead.
        if ($relativePath -match '^(\.venv|venv|weights|api_data|outputs|videos|__pycache__)/') {
            continue
        }
        Copy-SourceFile -RelativePath $relativePath
    }

    foreach ($relativePath in $requiredExtraFiles) {
        Copy-SourceFile -RelativePath $relativePath
    }

    if ($IncludeExperimentalTennisBallModel) {
        if ($Sport -ne 'tennis') {
            throw 'IncludeExperimentalTennisBallModel is only valid with -Sport tennis.'
        }
        Copy-SourceFile -RelativePath 'weights/yolo11s-ball.pt'
    }

    # The GPU package is intentionally created from the working tree: during
    # a staged multi-agent rollout, a newly added runtime module might not yet
    # be in Git's index.  Include only untracked *application-source* files
    # from the explicit service allow-list.  Tests, output artifacts, secrets,
    # weights and virtual environments remain excluded by construction.
    $deployableUntrackedPrefixes = @(
        'api/',
        'apps/',
        'badminton_analysis/',
        'business_gateway/',
        'good_badminton_contracts/',
        'webui/',
        'evaluation/performance/'
    )
    $untrackedFiles = & git -C $repoRoot ls-files --others --exclude-standard
    if ($LASTEXITCODE -ne 0) {
        throw 'git ls-files --others failed; cannot safely include new runtime source.'
    }
    foreach ($relativePath in $untrackedFiles) {
        $normalizedPath = $relativePath.Replace('\\', '/')
        $isDeployable = $deployableUntrackedPrefixes | Where-Object {
            $normalizedPath.StartsWith($_, [System.StringComparison]::OrdinalIgnoreCase)
        }
        if ($isDeployable -and $normalizedPath -match '\.(py|json|ya?ml)$') {
            Copy-SourceFile -RelativePath $normalizedPath
        }
    }
    if ($IncludeTrackNetABTools) {
        foreach ($relativePath in $trackNetABFiles) {
            Copy-SourceFile -RelativePath $relativePath
        }
    }

    $apiEntry = Join-Path $packageRoot 'api/app.py'
    $pureStreamEntry = Join-Path $packageRoot 'api/gpu_stream_app.py'
    $launcher = Join-Path $packageRoot 'deploy/start_gpu_api_container.sh'
    $sportLauncher = Join-Path $packageRoot 'deploy/start_sport_gpu_container.sh'
    $analysisPipeline = Join-Path $packageRoot 'webui/pipeline.py'
    if (-not (Test-Path -LiteralPath $apiEntry -PathType Leaf) -or -not (Test-Path -LiteralPath $pureStreamEntry -PathType Leaf) -or -not (Test-Path -LiteralPath $launcher -PathType Leaf) -or -not (Test-Path -LiteralPath $sportLauncher -PathType Leaf) -or -not (Test-Path -LiteralPath $analysisPipeline -PathType Leaf)) {
        throw 'Package validation failed: legacy API, pure stream API, launchers, or legacy pipeline is missing.'
    }

    $absoluteOutputPath = [System.IO.Path]::GetFullPath($OutputPath)
    $outputDirectory = Split-Path -Parent $absoluteOutputPath
    New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
    if (Test-Path -LiteralPath $absoluteOutputPath) {
        Remove-Item -LiteralPath $absoluteOutputPath -Force
    }
    # Compress-Archive writes Windows backslashes into ZIP entry names on some
    # PowerShell versions.  Linux unzip then treats them as literal characters
    # instead of directories.  Create entries explicitly with POSIX separators
    # so the same package is safe to unpack in the rented Ubuntu container.
    Add-Type -AssemblyName System.IO.Compression
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zip = [System.IO.Compression.ZipFile]::Open(
        $absoluteOutputPath,
        [System.IO.Compression.ZipArchiveMode]::Create
    )
    try {
        Get-ChildItem -LiteralPath $packageRoot -File -Recurse | ForEach-Object {
            # Windows PowerShell 5.1 targets .NET Framework and has no
            # System.IO.Path.GetRelativePath, so calculate it without relying
            # on PowerShell 7/.NET 6 APIs.
            $relativePath = $_.FullName.Substring($packageRoot.Length).TrimStart('\', '/').Replace('\', '/')
            $entryPath = "$packageName/$relativePath"
            [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
                $zip,
                $_.FullName,
                $entryPath,
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
    Write-Host "Upload it to /root/$packageName-upload.zip, then run:"
    Write-Host "bash /root/$packageName/deploy/refresh_gpu_api_from_zip.sh /root/$packageName-upload.zip /root/$packageName $Sport"
    if ($IncludeTrackNetABTools) {
        Write-Host 'TrackNet A/B tools are included; source code and checkpoint ZIPs remain separate uploads.'
    }
    if ($IncludeExperimentalTennisBallModel) {
        Write-Host 'Included yolo11s-ball.pt only as experimental tennis-ball evidence; provision tennis pose weights separately.'
    }
    Write-Host 'First deployment only (when that fixed directory does not yet exist):'
    Write-Host "unzip -p /root/$packageName-upload.zip $packageName/deploy/refresh_gpu_api_from_zip.sh | bash -s -- /root/$packageName-upload.zip /root/$packageName $Sport"
}
finally {
    if (Test-Path -LiteralPath $stagingRoot) {
        Remove-Item -LiteralPath $stagingRoot -Recurse -Force
    }
}
