#requires -version 5.1
$ErrorActionPreference = "Stop"
$TaskName = "GoodBadmintonVenueGateway"
$TaskPath = "\GoodBadminton\"
Stop-ScheduledTask -TaskName $TaskName -TaskPath $TaskPath -ErrorAction SilentlyContinue
Write-Host "Stopped $TaskName. It remains enabled for the next system startup."
