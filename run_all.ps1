param(
    [string]$RunDir = "runs/demo",
    [int]$Iterations = 1,
    [int]$Episodes = 1,
    [string]$Config = "configs/cpu.yaml"
)
$ErrorActionPreference = "Stop"
Push-Location -LiteralPath $PSScriptRoot
try {
    uv sync --locked
    if ($LASTEXITCODE -ne 0) { throw "Environment installation failed" }
    uv run pytest -q
    if ($LASTEXITCODE -ne 0) { throw "Tests failed" }
    uv run python -m rllib_cpu.train --config $Config --iterations $Iterations --output $RunDir
    if ($LASTEXITCODE -ne 0) { throw "Training failed" }
    uv run python -m rllib_cpu.evaluate --run $RunDir --episodes $Episodes
    if ($LASTEXITCODE -ne 0) { throw "Evaluation failed" }
    uv run python -m rllib_cpu.plot --run $RunDir
    if ($LASTEXITCODE -ne 0) { throw "Plotting failed" }
    Write-Host "Completed: $RunDir"
} finally {
    Pop-Location
}
