#requires -version 5.1
[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$TaskName = "GoodBadmintonVenueGateway"
$TaskPath = "\GoodBadminton\"
Stop-ScheduledTask -TaskName $TaskName -TaskPath $TaskPath -ErrorAction SilentlyContinue
Unregister-ScheduledTask -TaskName $TaskName -TaskPath $TaskPath -Confirm:$false -ErrorAction SilentlyContinue
Write-Host "Removed the scheduled task. Program files, configuration, logs, and any currently present spool files were preserved."
