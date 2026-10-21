[CmdletBinding()]
param(
    [string]$Output = "outputs/validation-replay-seed42",
    [switch]$ReportOnly
)
$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot
try {
    if (-not $ReportOnly) {
        foreach ($policy in @("random", "validation")) {
            & python run_validation_replay.py run --output $Output --policy $policy
            if ($LASTEXITCODE -ne 0) { throw "Validation-replay $policy run stopped. Rerun this command to resume committed checkpoints." }
        }
    }
    & python run_validation_replay.py report --output $Output
    if ($LASTEXITCODE -ne 0) { throw "Comparison report failed. Use -ReportOnly to retry without training." }
}
finally { Pop-Location }
