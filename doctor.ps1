Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$TtsPython = Join-Path $Root "runtime\tts\.venv\Scripts\python.exe"
$TtsRepo = Join-Path $Root "runtime\tts\Chatterbox-Finnish"
$RvcPython = Join-Path $Root "runtime\rvc\.venv\Scripts\python.exe"
$RvcRepo = Join-Path $Root "runtime\rvc\RVC-WebUI"

$checks = @(
    @("TTS Python", $TtsPython),
    @("Finnish fine-tune", (Join-Path $TtsRepo "models\best_finnish_multilingual_cp986.safetensors")),
    @("Finnish reference", (Join-Path $TtsRepo "samples\reference_finnish.wav")),
    @("Chatterbox base VE", (Join-Path $TtsRepo "pretrained_models\ve.safetensors")),
    @("RVC Python", $RvcPython),
    @("HuBERT", (Join-Path $RvcRepo "assets\hubert_base\config.json")),
    @("RMVPE", (Join-Path $RvcRepo "assets\rmvpe\rmvpe.pt")),
    @("RVC pretrained G40k", (Join-Path $RvcRepo "assets\pretrained_v2\f0G40k.pth")),
    @("RVC pretrained D40k", (Join-Path $RvcRepo "assets\pretrained_v2\f0D40k.pth")),
    @("Mute assets", (Join-Path $RvcRepo "logs\mute"))
)

$failed = $false
Write-Host "Zundanen - doctor" -ForegroundColor Cyan
foreach ($check in $checks) {
    $name = $check[0]
    $path = $check[1]
    if (Test-Path $path) {
        Write-Host "[OK]   $name" -ForegroundColor Green
    } else {
        Write-Host "[MISS] $name -> $path" -ForegroundColor Red
        $failed = $true
    }
}

if (Test-Path $TtsPython) {
    & $TtsPython -c "import torch, flask, soundfile, safetensors, webview; print('[OK] TTS/Desktop imports; CUDA:', torch.cuda.is_available())"
    if ($LASTEXITCODE -ne 0) { $failed = $true }
}
if (Test-Path $RvcPython) {
    & $RvcPython -c "import torch, faiss, librosa, numpy; print('[OK] RVC imports; CUDA:', torch.cuda.is_available())"
    if ($LASTEXITCODE -ne 0) { $failed = $true }
}

$Docker = Get-Command docker -ErrorAction SilentlyContinue
if ($null -ne $Docker) {
    try {
        docker info --format "{{.ServerVersion}}" *> $null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[OK]   Docker (optional Accent support)" -ForegroundColor Green
        } else {
            Write-Host "[OPT]  Docker is installed but not running. Start Docker Desktop to use Accent editing." -ForegroundColor Yellow
        }
    } catch {
        Write-Host "[OPT]  Docker could not be checked. Accent support is optional." -ForegroundColor Yellow
    }
} else {
    Write-Host "[OPT]  Docker Desktop not found. Required only for optional Accent support." -ForegroundColor Yellow
}

if ($failed) {
    Write-Host "Doctor found problems." -ForegroundColor Red
    exit 1
}
Write-Host "Doctor: all required files and imports look OK." -ForegroundColor Green
