#requires -version 5.1
[CmdletBinding()]
param(
    [string]$ConfigFile = "$env:ProgramData\GoodBadminton\venue-gateway.env"
)

$ErrorActionPreference = "Stop"
$TaskName = "GoodBadmintonVenueGateway"
$TaskPath = "\GoodBadminton\"
if (-not (Test-Path $ConfigFile)) { throw "Configuration file does not exist: $ConfigFile" }
if (Select-String -Quiet -SimpleMatch "SET_BY_" $ConfigFile) { throw "Configuration still contains SET_BY_ placeholders." }
Enable-ScheduledTask -TaskName $TaskName -TaskPath $TaskPath | Out-Null
Start-ScheduledTask -TaskName $TaskName -TaskPath $TaskPath
Write-Host "Started $TaskName."
