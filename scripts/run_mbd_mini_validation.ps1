param(
  [int]$TopK = 2,
  [int]$MinUserPositiveEvents = 2,
  [int]$MaxUsers = 0,
  [int]$Factors = 3,
  [int]$Neighbors = 3,
  [switch]$ForceDownload,
  [switch]$ForceExtract
)

$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $py)) {
  throw "Python virtual environment was not found at $py"
}

Push-Location $root

$argsList = @(
  "-m", "src.pipelines.run_mbd_mini_validation",
  "--top-k", $TopK,
  "--min-user-positive-events", $MinUserPositiveEvents,
  "--max-users", $MaxUsers,
  "--n-factors", $Factors,
  "--n-neighbors", $Neighbors
)

if ($ForceDownload) {
  $argsList += "--force-download"
}
if ($ForceExtract) {
  $argsList += "--force-extract"
}

& $py @argsList
if ($LASTEXITCODE -ne 0) { $code = $LASTEXITCODE; Pop-Location; exit $code }

Pop-Location

Write-Host '[ok] MBD-mini banking validation completed'
