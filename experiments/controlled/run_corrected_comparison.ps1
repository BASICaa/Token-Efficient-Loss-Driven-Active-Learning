[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Dataset,
    [Parameter(Mandatory = $true)][string]$Model,
    [ValidateRange(10, 100)][int]$Examples = 100,
    [string]$Output = "outputs/corrected-comparison-100",
    [switch]$FullValidation,
    [switch]$PrepareOnly
)

$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot
try {
    $datasetPath = (Resolve-Path -LiteralPath $Dataset).Path
    $modelPath = (Resolve-Path -LiteralPath $Model).Path
    $conditions = @(
        @{ Name = "dolly"; Mode = "dolly"; Method = "dolly_baseline"; Bank = $null },
        @{ Name = "local-loss"; Mode = "local-loss"; Method = "tefld_dataset_loss"; Bank = $null },
        @{ Name = "local-validation"; Mode = "local-validation"; Method = "tefld_dataset_validation"; Bank = $null },
        @{ Name = "sol61-validation-bank"; Mode = "api-validation"; Method = "tefld_validation"; Bank = "outputs/banks/sol61-validation.json" },
        @{ Name = "sol61-loss-bank"; Mode = "api-loss"; Method = "tefld_loss"; Bank = "outputs/banks/sol61-loss.json" },
        @{ Name = "luna-validation-bank"; Mode = "api-validation"; Method = "tefld_validation"; Bank = "outputs/banks/luna-validation.json" }
    )
    foreach ($condition in $conditions) {
        if ($condition.Bank -and -not (Test-Path -LiteralPath $condition.Bank -PathType Leaf)) {
            throw "Missing saved bank: $($condition.Bank). Export existing admissions first; no generation fallback is enabled."
        }
    }
    # Prepare every condition before starting any training. No paid option is
    # passed anywhere in this launcher, including API-derived conditions.
    foreach ($condition in $conditions) {
        $runOutput = Join-Path $Output $condition.Name
        $prepareArgs = @("run_controlled.py", "prepare", "--mode", $condition.Mode,
            "--dataset", $datasetPath, "--model", $modelPath,
            "--examples", "$Examples", "--output", $runOutput)
        if ($FullValidation) { $prepareArgs += "--full-validation" }
        & python @prepareArgs
        if ($LASTEXITCODE -ne 0) { throw "Preparation failed: $($condition.Name). No subsequent condition was started." }
    }
    if ($PrepareOnly) {
        Write-Output "All six conditions prepared. No training or generation started."
        return
    }
    foreach ($condition in $conditions) {
        $runOutput = Join-Path $Output $condition.Name
        $runArgs = @("run_controlled.py", "run", "--output", $runOutput)
        if ($condition.Bank) { $runArgs += @("--candidate-bank", $condition.Bank) }
        & python @runArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Stopped at $($condition.Name). Completed checkpoints are preserved. A fixed bank may lack category coverage; no new paid generation will be attempted."
        }
    }
    $rows = @(
        foreach ($condition in $conditions) {
            $runOutput = Join-Path $Output $condition.Name
            $statePath = Join-Path $runOutput "arms/$($condition.Mode)/pilot/$($condition.Method)_seed_42/state.json"
            $state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
            $evaluationPath = Join-Path $state.final_checkpoint.path "validation.json"
            $evaluation = Get-Content -LiteralPath $evaluationPath -Raw | ConvertFrom-Json
            [pscustomobject]@{
                Condition = $condition.Name
                Status = $state.status
                UniqueExamples = @($state.ledger).Count
                EvaluationExamples = @($evaluation.rows).Count
                MacroNLL = $evaluation.macro_nll
                TokenWeightedNLL = $evaluation.token_weighted_nll
                TrainingTokenExposures = $state.tokens
                OptimizerUpdates = @($state.losses | Group-Object -Property round, tokens_after_group).Count
            }
        }
    )
    $report = [ordered]@{
        Scope = "Fresh corrected-branch comparison. Fixed-bank API conditions measure selection over existing synthetic data, not live adaptive generation."
        Seed = 42
        TrainingExamplesPerCondition = $Examples
        FullValidation = [bool]$FullValidation
        NewProviderCalls = 0
        FinalTestScored = $false
        Results = $rows
    }
    $reportPath = Join-Path $Output "comparison.json"
    $report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding utf8
    $rows | Sort-Object MacroNLL | Format-Table -AutoSize
    Write-Output "Saved comparison: $reportPath"
}
finally {
    Pop-Location
}
