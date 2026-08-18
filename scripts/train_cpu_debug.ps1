# CPU smoke-test training for the Depth Anything integration.
# Run from the project root:
#   .\scripts\train_cpu_debug.ps1                        # DA V1 ViT-B (default)
#   .\scripts\train_cpu_debug.ps1 -Experiment layered_re10k_cpu_debug      # DA V2
#   .\scripts\train_cpu_debug.ps1 -Encoder vits                            # DA V1 ViT-S
param(
    [string]$DataPath = "data/RealEstate10K",
    [int]$Epochs = 1,
    [string]$Experiment = "layered_re10k_cpu_debug_v1",
    [string]$Encoder = ""
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$overrides = @(
    "train.py",
    "+experiment=$Experiment",
    "dataset.data_path=$DataPath",
    "optimiser.num_epochs=$Epochs"
)
if ($Encoder -ne "") {
    $overrides += "model.depth.encoder=$Encoder"
}
python @overrides
