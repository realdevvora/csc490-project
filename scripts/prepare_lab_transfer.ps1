param(
    [string]$DataRoot = 'D:\csc409-data\d2c_v3',
    [switch]$SkipData
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$outputRoot = Join-Path $projectRoot 'outputs\lab-transfer'
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null

Push-Location $projectRoot
try {
    $branch = (git branch --show-current).Trim()
    if (-not $branch) { throw 'A named branch is required for the transfer bundle.' }
    git bundle create (Join-Path $outputRoot 'repo.bundle') $branch
    if ($LASTEXITCODE -ne 0) { throw 'Git bundle creation failed.' }

    # Overlay current files, including uncommitted experiments, on the bundled clone.
    # Explicit paths keep local environments, credentials, model weights and logs out.
    $sourcePaths = @(
        'README.md', 'pyproject.toml', '.gitignore', 'AGENTS.md',
        'src', 'scripts', 'tests', 'notebooks', 'models', 'configs', 'docs',
        'data/README.md', 'data/splits', 'data/scripts',
        'outputs/dino_fit/sample_manifest.json', 'outputs/dino_fit/dinov2',
        'outputs/dino_fit/dinov3'
    )
    $sourcePaths = @($sourcePaths | Where-Object { Test-Path -LiteralPath $_ })
    tar -czf (Join-Path $outputRoot 'working-tree.tar.gz') '--exclude=*/__pycache__' '--exclude=*.egg-info' '--exclude=*/runs' '--exclude=*/.venv' '--exclude=*/.ipynb_checkpoints' @sourcePaths
    if ($LASTEXITCODE -ne 0) { throw 'Working-tree archive creation failed.' }

    if (-not $SkipData) {
        if (-not (Test-Path -LiteralPath (Join-Path $DataRoot 'master_dataset_with_benign.json'))) {
            throw "Dataset metadata not found under $DataRoot"
        }
        $dataDirectory = Get-Item -LiteralPath $DataRoot
        # PNG screenshots are already compressed: tar avoids a long recompression pass.
        tar -cf (Join-Path $outputRoot 'dataset.tar') -C $dataDirectory.Parent.FullName $dataDirectory.Name
        if ($LASTEXITCODE -ne 0) { throw 'Dataset archive creation failed.' }
    }

    Get-ChildItem -LiteralPath $outputRoot -File | Select-Object Name, Length
    Write-Host "Transfer prepared for branch $branch under $outputRoot"
}
finally {
    Pop-Location
}
