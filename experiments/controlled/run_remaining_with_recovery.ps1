[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][ValidateRange(1, 900000)][double]$TeacherBudgetToman,
    [string]$Source = "outputs/corrected-comparison-100",
    [string]$Output = "outputs/corrected-recovery-900k",
    [switch]$PrepareOnly
)

$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot
try {
    if (-not $PrepareOnly -and [string]::IsNullOrWhiteSpace($env:HIBANA_API_KEY)) {
        throw "HIBANA_API_KEY is missing in this PowerShell session. Set it locally before running recovery; do not paste it into chat. No training or API calls were started."
    }
    $conditions = @(
        @{Name="sol61-loss-bank"; Bank="outputs/banks/sol61-loss.json"; Profile=$null},
        @{Name="luna-validation-bank"; Bank="outputs/banks/luna-validation.json"; Profile="experiment_design_2026-10-18/recovery_luna_generation.json"}
    )
    foreach ($condition in $conditions) {
        $sourcePath = Join-Path $Source $condition.Name
        $outputPath = Join-Path $Output $condition.Name
        $sourceStates = @(Get-ChildItem -LiteralPath $sourcePath -Filter state.json -Recurse)
        if ($sourceStates.Count -gt 1) { throw "Ambiguous source state: $sourcePath" }
        if ($sourceStates.Count -eq 1) {
            $sourceState = Get-Content -LiteralPath $sourceStates[0].FullName -Raw | ConvertFrom-Json
            if ($sourceState.status -eq "complete") {
                Write-Output "Already complete: $($condition.Name). Keeping original result."
                continue
            }
        }
        if (-not (Test-Path -LiteralPath (Join-Path $outputPath "recovery_plan.json"))) {
            $prepareArgs = @("run_bank_recovery.py", "prepare", "--source", $sourcePath,
                "--output", $outputPath, "--candidate-bank", $condition.Bank, "--budget-root", $Output)
            if ($condition.Profile) { $prepareArgs += @("--generation-protocol", $condition.Profile) }
            & python @prepareArgs
            if ($LASTEXITCODE -ne 0) { throw "Recovery preparation failed: $($condition.Name)." }
        }
        if (-not $PrepareOnly) {
            & python run_bank_recovery.py run --output $outputPath --allow-paid-api --teacher-budget-toman $TeacherBudgetToman
            if ($LASTEXITCODE -ne 0) {
                throw "Recovery stopped: $($condition.Name). Check the error; accepted candidates and committed rounds are preserved. Rerun with the same settings to resume."
            }
        }
    }
    if ($PrepareOnly) {
        Write-Output "Recovery prepared. No API calls or training started."
        return
    }
    $names = @("dolly", "local-loss", "local-validation", "sol61-validation-bank", "sol61-loss-bank", "luna-validation-bank")
    $rows = @(
        foreach ($name in $names) {
            $resultRoot = Join-Path $Output $name
            $states = @(Get-ChildItem -LiteralPath $resultRoot -Filter state.json -Recurse -ErrorAction SilentlyContinue)
            if (-not $states.Count) {
                $resultRoot = Join-Path $Source $name
                $states = @(Get-ChildItem -LiteralPath $resultRoot -Filter state.json -Recurse)
            }
            if ($states.Count -ne 1) { throw "Expected one state for $name." }
            $state = Get-Content -LiteralPath $states[0].FullName -Raw | ConvertFrom-Json
            if ($state.status -ne "complete") { throw "Incomplete condition: $name." }
            $evaluation = Get-Content -LiteralPath (Join-Path $state.final_checkpoint.path "validation.json") -Raw | ConvertFrom-Json
            $recoveryResult = Join-Path $resultRoot "recovery_result.json"
            $newCandidates = 0
            if (Test-Path -LiteralPath $recoveryResult) {
                $newCandidates = (Get-Content -LiteralPath $recoveryResult -Raw | ConvertFrom-Json).new_verified_candidates
            }
            [pscustomobject]@{
                Condition=$name; UniqueExamples=@($state.ledger).Count; EvaluationExamples=@($evaluation.rows).Count
                MacroNLL=$evaluation.macro_nll; TokenWeightedNLL=$evaluation.token_weighted_nll
                TrainingTokenExposures=$state.tokens; NewVerifiedCandidates=$newCandidates; State=$states[0].FullName
            }
        }
    )
    [ordered]@{Scope="Original completed runs plus separately identified bank-replenishment continuations.";
        SharedRecoveryBudgetToman=$TeacherBudgetToman; FinalTestScored=$false; Results=$rows} |
        ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $Output "comparison.json") -Encoding utf8
    $rows | Sort-Object MacroNLL | Format-Table -AutoSize
    Write-Output "Combined results: $(Join-Path $Output 'comparison.json')"
}
finally {
    Pop-Location
}
