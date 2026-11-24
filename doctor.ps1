Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$TtsPython = Join-Path $Root "runtime\tts\.venv\Scripts\python.exe"
$RvcPython = Join-Path $Root "runtime\rvc\.venv\Scripts\python.exe"
$RvcRepo = Join-Path $Root "runtime\rvc\RVC-WebUI"
$NativeProbe = Join-Path $Root "aalto_native_probe.py"

$checks = @(
    @("TTS Python", $TtsPython),
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

$nativeReady = $false
if ((Test-Path $NativeProbe) -and (Test-Path $RvcPython)) {
    & $RvcPython $NativeProbe *> $null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK]   Native Windows Aalto/Kaldi (Accent)" -ForegroundColor Green
        $nativeReady = $true
    } else {
        Write-Host "[MISS] Native Windows Aalto/Kaldi Accent engine is not ready. Run setup.bat again." -ForegroundColor Red
        $failed = $true
    }

    & $RvcPython -c "import pyworld" *> $null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK]   PyWORLD (Accent)" -ForegroundColor Green
    } else {
        Write-Host "[MISS] PyWORLD is missing. Run setup.bat again." -ForegroundColor Red
        $failed = $true
    }
} else {
    Write-Host "[MISS] Accent probe or RVC Python is missing. Run setup.bat again." -ForegroundColor Red
    $failed = $true
}

if ($failed) {
    Write-Host "Doctor found problems." -ForegroundColor Red
    exit 1
}
Write-Host "Doctor: all required files and imports look OK." -ForegroundColor Green
