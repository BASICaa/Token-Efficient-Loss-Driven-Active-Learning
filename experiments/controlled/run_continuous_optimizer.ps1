[CmdletBinding()]
param(
    [string]$Output = "outputs/optimizer-continuous-seed42",
    [switch]$ReportOnly
)

$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot
try {
    $plan = Get-Content -LiteralPath (Join-Path $Output "plan.json") -Raw | ConvertFrom-Json
    if ($plan.settings.optimizer_lifecycle -ne "continuous" -or
        $plan.optimizer_comparison.planned_new_runs -ne 1 -or
        $plan.optimizer_comparison.planned_selection -ne "random" -or
        $plan.optimizer_comparison.planned_replay -ne "off" -or
        @($plan.settings.seeds).Count -ne 1) { throw "Not a prepared single-run optimizer diagnostic." }
    if (-not $ReportOnly) {
        $seed = $plan.settings.seeds[0]
        & python run_selection_diagnostic.py run --output $Output --selection random --replay off --seed $seed
        if ($LASTEXITCODE -ne 0) { throw "Optimizer diagnostic stopped. Committed model and optimizer state are preserved; rerun this command to resume." }
    }
    & python prepare_optimizer_study.py report --output $Output
    if ($LASTEXITCODE -ne 0) { throw "Optimizer report failed. Use -ReportOnly to retry without training." }
}
finally {
    Pop-Location
}
