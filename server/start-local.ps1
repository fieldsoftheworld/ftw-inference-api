# Free port 8080 (stale uvicorn/python often holds it) and start the API once.
$Port = 8080
$ErrorActionPreference = "SilentlyContinue"
Get-NetTCPConnection -LocalPort $Port -State Listen | ForEach-Object {
    Stop-Process -Id $_.OwningProcess -Force
}
Start-Sleep -Seconds 1
if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
    Write-Error "Port $Port is still in use. End remaining Python processes in Task Manager, then retry."
    exit 1
}

$env:AWS_ACCESS_KEY_ID = "test"
$env:AWS_SECRET_ACCESS_KEY = "test"
Set-Location $PSScriptRoot
Write-Host "Starting uvicorn on http://127.0.0.1:$Port ..."
py -3.11 -m uvicorn app.main:app --host 127.0.0.1 --port $Port
