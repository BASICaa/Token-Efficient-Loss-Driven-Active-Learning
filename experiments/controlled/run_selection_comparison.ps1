[CmdletBinding()]
param(
    [string]$Output = "outputs/selection-diagnostic-dolly",
    [switch]$ReportOnly
)

$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot
try {
    if ($ReportOnly) {
        & python run_selection_diagnostic.py report --output $Output
    } else {
        # Explicit user invocation starts local training. No API flags exist.
        & python run_selection_diagnostic.py run --output $Output
    }
    if ($LASTEXITCODE -ne 0) { throw "Selection diagnostic stopped. See the error and any last_failure.json; committed checkpoints are preserved." }
}
finally {
    Pop-Location
}
