[CmdletBinding()]
param(
    [string]$Output = "outputs/selection-diagnostic-dolly",
    [switch]$ReportOnly,
    [switch]$StatusOnly
)

$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot
try {
    if ($ReportOnly -and $StatusOnly) { throw "Choose ReportOnly or StatusOnly." }
    $planPath = Join-Path $Output "plan.json"
    $plan = Get-Content -LiteralPath $planPath -Raw | ConvertFrom-Json
    # .NET hashing also works when a Conda child shell has a restricted
    # PowerShell module path and cannot autoload Get-FileHash.
    $planStream = [System.IO.File]::OpenRead((Resolve-Path -LiteralPath $planPath).Path)
    $hasher = [System.Security.Cryptography.SHA256]::Create()
    try {
        $planHash = [System.BitConverter]::ToString($hasher.ComputeHash($planStream)).Replace("-", "").ToLowerInvariant()
    } finally {
        $planStream.Dispose()
        $hasher.Dispose()
    }
    $jobs = @(
        foreach ($seed in $plan.settings.seeds) {
            foreach ($replay in $plan.settings.replays) {
                foreach ($selection in @("random", "loss", "validation")) {
                    $statePath = Join-Path $Output "seed_$seed/${selection}__replay_$replay/state.json"
                    $state = $null
                    if (Test-Path -LiteralPath $statePath) {
                        $state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
                        if ($state.identity.plan_sha256 -ne $planHash -or
                            $state.identity.seed -ne $seed -or
                            $state.identity.selection -ne $selection -or
                            $state.identity.replay -ne $replay) {
                            throw "Saved run identity differs from its plan: $statePath"
                        }
                        if ($state.status -eq "complete" -and
                            ($state.update -ne $plan.settings.updates -or
                             $state.tokens -ne $plan.tokens_by_seed.([string]$seed))) {
                            throw "Completed run violates its resource budget: $statePath"
                        }
                    }
                    [pscustomobject]@{
                        Seed=$seed; Selection=$selection; Replay=$replay
                        Status=if ($state) {$state.status} else {"not started"}
                        Updates=if ($state) {$state.update} else {0}
                    }
                }
            }
        }
    )
    $pending = @($jobs | Where-Object {$_.Status -ne "complete"})
    Write-Output "Runs: $($jobs.Count - $pending.Count)/$($jobs.Count) complete; $($pending.Count) remaining."
    if ($StatusOnly) {
        $jobs | Format-Table -AutoSize
        return
    }
    if ($ReportOnly) {
        & python run_selection_diagnostic.py report --output $Output
        if ($LASTEXITCODE -ne 0) { throw "Diagnostic reporting failed; training was not started." }
    } else {
        foreach ($job in $pending) {
            Write-Output "Starting $($job.Selection), replay=$($job.Replay), seed=$($job.Seed), saved updates=$($job.Updates)/$($plan.settings.updates)."
            # Each completed child process releases all of its CUDA context,
            # model, optimizer and allocator state before the next model loads.
            # The fingerprinted training implementation and plan stay unchanged.
            & python run_selection_diagnostic.py run --output $Output --selection $job.Selection --replay $job.Replay --seed $job.Seed
            if ($LASTEXITCODE -ne 0) {
                throw "Stopped at $($job.Selection), replay=$($job.Replay), seed=$($job.Seed). Committed checkpoints are preserved. For CUDA out-of-memory errors, free GPU memory and rerun this command; no completed run will be repeated."
            }
        }
        & python run_selection_diagnostic.py report --output $Output
        if ($LASTEXITCODE -ne 0) { throw "Training runs finished, but reporting failed. Use -ReportOnly to retry reporting." }
    }
}
finally {
    Pop-Location
}
