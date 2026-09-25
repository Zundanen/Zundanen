param(
    [switch]$CI,
    [switch]$Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
# Keep console/native-tool text in UTF-8 where supported.
try {
    [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
    $OutputEncoding = [Console]::OutputEncoding
}
catch {}

$ProgressPreference = "SilentlyContinue"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Runtime = Join-Path $Root "runtime"
$TtsRoot = Join-Path $Runtime "tts"
$TtsRepo = Join-Path $TtsRoot "Chatterbox-Finnish"
$TtsVenv = Join-Path $TtsRoot ".venv"
$RvcRoot = Join-Path $Runtime "rvc"
$RvcRepo = Join-Path $RvcRoot "RVC-WebUI"
$RvcVenv = Join-Path $RvcRoot ".venv"
$Exports = Join-Path $RvcRoot "exports"

function Write-Step([string]$Text) {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor DarkGray
    Write-Host $Text -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor DarkGray
}

function Refresh-Path {
    $machine = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $user = [Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = "$machine;$user"
}

function Invoke-Native {
    param(
        [Parameter(Mandatory=$true)][string]$File,
        [Parameter(Mandatory=$true)][string[]]$Arguments,
        [string]$WorkingDirectory = ""
    )
    if ($WorkingDirectory) { Push-Location $WorkingDirectory }
    try {
        Write-Host "> $File $($Arguments -join ' ')" -ForegroundColor DarkGray
        & $File @Arguments
        $code = $LASTEXITCODE
        if ($code -ne 0) {
            throw "Command failed with exit code ${code}: $File"
        }
    }
    finally {
        if ($WorkingDirectory) { Pop-Location }
    }
}

function Ensure-WingetPackage([string]$Id) {
    Write-Host "Checking $Id ..."

    # winget localizes its own console output to the Windows display language.
    # Keep that raw output hidden so the Zundanen setup log stays
    # consistently English on Japanese, Finnish, English, and other systems.
    & winget install --id $Id -e --source winget --accept-package-agreements --accept-source-agreements --silent --disable-interactivity *> $null
    $installCode = $LASTEXITCODE

    if ($installCode -eq 0) {
        Write-Host "[OK] $Id installed or updated." -ForegroundColor Green
        Refresh-Path
        return
    }

    # A common non-zero result means the package is already installed and
    # there is no newer version. Verify it quietly instead of showing the
    # localized winget table.
    & winget list --id $Id -e --accept-source-agreements --disable-interactivity *> $null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] $Id already installed." -ForegroundColor Green
        Refresh-Path
        return
    }

    throw "Could not install $Id (winget exit code $installCode)."
}

function Test-VCRedistX64 {
    # Microsoft Visual C++ 2015-2022 x64 runtime.
    # Check both the official registry marker and the DLLs commonly required
    # by Windows PyTorch wheels. This avoids reinstalling the runtime on PCs
    # where it is already usable, while still fixing clean Windows installs.
    $regPath = "HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64"
    $installed = $false
    $version = ""

    if (Test-Path $regPath) {
        try {
            $runtime = Get-ItemProperty -Path $regPath -ErrorAction Stop
            $installed = ($runtime.Installed -eq 1)
            if ($runtime.Version) {
                $version = [string]$runtime.Version
            }
        }
        catch {
            $installed = $false
        }
    }

    $requiredDlls = @(
        (Join-Path $env:WINDIR "System32\vcruntime140.dll"),
        (Join-Path $env:WINDIR "System32\vcruntime140_1.dll"),
        (Join-Path $env:WINDIR "System32\msvcp140.dll")
    )
    $dllsPresent = @($requiredDlls | Where-Object { -not (Test-Path $_) }).Count -eq 0

    if ($installed -and $dllsPresent) {
        if ($version) {
            Write-Host "[OK] Microsoft Visual C++ Runtime x64 already installed ($version)." -ForegroundColor Green
        }
        else {
            Write-Host "[OK] Microsoft Visual C++ Runtime x64 already installed." -ForegroundColor Green
        }
        return $true
    }

    return $false
}

function Ensure-VCRedistX64 {
    if (Test-VCRedistX64) {
        return
    }

    Write-Host "Microsoft Visual C++ Runtime x64 is missing or incomplete. Installing..."
    Ensure-WingetPackage "Microsoft.VCRedist.2015+.x64"

    if (-not (Test-VCRedistX64)) {
        throw "Microsoft Visual C++ Runtime installation completed, but required runtime DLLs were not detected. Restart Windows and run setup.bat again."
    }
}


function Test-WebView2Runtime {
    # Microsoft-documented WebView2 Evergreen Runtime detection.
    $guid = "{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"
    $paths = @(
        "HKLM:\SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\$guid",
        "HKCU:\Software\Microsoft\EdgeUpdate\Clients\$guid"
    )

    foreach ($path in $paths) {
        if (-not (Test-Path $path)) { continue }
        try {
            $pv = [string](Get-ItemPropertyValue -Path $path -Name "pv" -ErrorAction Stop)
            if ($pv -and $pv -ne "0.0.0.0") {
                Write-Host "[OK] Microsoft Edge WebView2 Runtime already installed ($pv)." -ForegroundColor Green
                return $true
            }
        }
        catch {}
    }
    return $false
}

function Ensure-WebView2Runtime {
    if (Test-WebView2Runtime) {
        return
    }

    Write-Host "Microsoft Edge WebView2 Runtime is missing. Installing..."
    Ensure-WingetPackage "Microsoft.EdgeWebView2Runtime"

    if (-not (Test-WebView2Runtime)) {
        throw "WebView2 Runtime installation completed, but the runtime was not detected. Restart Windows and run setup.bat again."
    }
}

function Resolve-Python([string]$Version, [string]$Tag) {
    Refresh-Path
    $py = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($py) {
        $out = & $py.Source "-$Version" -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $out) {
            return (($out | Select-Object -Last 1).ToString().Trim())
        }
    }

    $candidates = @(
        (Join-Path $env:LOCALAPPDATA "Programs\Python\Python$Tag\python.exe"),
        (Join-Path $env:LOCALAPPDATA "Python\pythoncore-$Version-64\python.exe"),
        (Join-Path $env:ProgramFiles "Python$Tag\python.exe")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) { return $candidate }
    }

    $roots = @(
        (Join-Path $env:LOCALAPPDATA "Programs\Python"),
        (Join-Path $env:LOCALAPPDATA "Python")
    )
    foreach ($searchRoot in $roots) {
        if (-not (Test-Path $searchRoot)) { continue }
        foreach ($candidate in Get-ChildItem $searchRoot -Filter python.exe -Recurse -ErrorAction SilentlyContinue) {
            $reported = & $candidate.FullName -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
            if ($LASTEXITCODE -eq 0 -and $reported -eq $Version) {
                return $candidate.FullName
            }
        }
    }
    throw "Python $Version x64 was installed but could not be located. Restart Windows and run setup.bat again."
}

function Resolve-Git {
    Refresh-Path
    $git = Get-Command git.exe -ErrorAction SilentlyContinue
    if ($git) { return $git.Source }
    $candidates = @(
        (Join-Path $env:ProgramFiles "Git\cmd\git.exe"),
        (Join-Path $env:LOCALAPPDATA "Programs\Git\cmd\git.exe")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) { return $candidate }
    }
    throw "Git was installed but could not be located. Restart Windows and run setup.bat again."
}

function Get-GpuInfo {
    Refresh-Path
    $smiCommand = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
    $smiPath = if ($smiCommand) { $smiCommand.Source } else { "" }
    if (-not $smiPath -and (Test-Path "$env:WINDIR\System32\nvidia-smi.exe")) {
        $smiPath = "$env:WINDIR\System32\nvidia-smi.exe"
    }
    if (-not $smiPath) {
        return [pscustomobject]@{ Mode="cpu"; Name="No NVIDIA GPU detected"; Capability=0.0 }
    }

    try {
        $rows = & $smiPath --query-gpu=name,compute_cap --format=csv,noheader 2>$null
        if ($LASTEXITCODE -eq 0 -and $rows) {
            $row = ($rows | Select-Object -First 1).ToString()
            $parts = $row -split ','
            $name = $parts[0].Trim()
            $cap = 0.0
            [double]::TryParse($parts[-1].Trim(), [Globalization.NumberStyles]::Float, [Globalization.CultureInfo]::InvariantCulture, [ref]$cap) | Out-Null
            $mode = if ($cap -ge 12.0) { "blackwell" } else { "nvidia" }
            return [pscustomobject]@{ Mode=$mode; Name=$name; Capability=$cap }
        }
    } catch {}

    $nameRows = & $smiPath --query-gpu=name --format=csv,noheader 2>$null
    $name = if ($nameRows) { ($nameRows | Select-Object -First 1).ToString().Trim() } else { "NVIDIA GPU" }
    $mode = if ($name -match 'RTX\s*50') { "blackwell" } else { "nvidia" }
    return [pscustomobject]@{ Mode=$mode; Name=$name; Capability=0.0 }
}


$script:SetupSleepPreventionActive = $false

function Start-SetupSleepPrevention {
    if ($env:OS -ne "Windows_NT") { return }

    try {
        if (-not ("ZundanenPowerState" -as [type])) {
            Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;

public static class ZundanenPowerState
{
    [DllImport("kernel32.dll")]
    public static extern uint SetThreadExecutionState(uint esFlags);
}
"@
        }

        # ES_CONTINUOUS (0x80000000) + ES_SYSTEM_REQUIRED (0x00000001)
        # Prevent system sleep while setup runs, but allow the display to turn off.
        $result = [ZundanenPowerState]::SetThreadExecutionState([uint32]2147483649)
        if ($result -eq 0) {
            Write-Warning "Could not enable temporary sleep prevention. Setup will continue."
            return
        }

        $script:SetupSleepPreventionActive = $true
        Write-Host "[OK] Automatic system sleep is temporarily disabled during setup." -ForegroundColor Green
    }
    catch {
        Write-Warning "Could not enable temporary sleep prevention: $($_.Exception.Message)"
    }
}

function Stop-SetupSleepPrevention {
    if (-not $script:SetupSleepPreventionActive) { return }

    try {
        # ES_CONTINUOUS only: restore the thread's normal execution-state request.
        [ZundanenPowerState]::SetThreadExecutionState([uint32]2147483648) | Out-Null
        Write-Host "[OK] Automatic system sleep settings restored." -ForegroundColor Green
    }
    catch {
        Write-Warning "Could not restore the temporary sleep-prevention state: $($_.Exception.Message)"
    }
    finally {
        $script:SetupSleepPreventionActive = $false
    }
}

if ($CI) {
    Write-Step "CI smoke setup"
    $python = (Get-Command python.exe -ErrorAction Stop).Source
    $ciVenv = Join-Path $Root ".ci-venv"
    if (Test-Path $ciVenv) { Remove-Item $ciVenv -Recurse -Force }
    Invoke-Native $python @("-m", "venv", $ciVenv)
    $ciPy = Join-Path $ciVenv "Scripts\python.exe"
    Invoke-Native $ciPy @("-m", "pip", "install", "--disable-pip-version-check", "flask>=3.0,<4")
    Invoke-Native $ciPy @("-m", "py_compile", (Join-Path $Root "app.py"), (Join-Path $Root "tts_worker.py"), (Join-Path $Root "trainer_worker.py"), (Join-Path $Root "desktop_launcher.py"))
    Write-Host "CI smoke setup OK" -ForegroundColor Green
    exit 0
}

Start-SetupSleepPrevention

try {
# Zundanen v1.0.0 setup
Write-Host "Zundanen v1.0.0 - clean Windows setup" -ForegroundColor Green
Write-Host "This creates isolated TTS and RVC environments under: $Runtime"
Write-Host "Large model files and PyTorch packages will be downloaded."

# Some Python wheels (notably ONNX) contain very long internal test paths.
# Classic Windows path limits can still affect extraction/install tools even on
# current Windows versions, so fail early with a useful message instead of a
# cryptic WinError 206 halfway through pip install.
$MaxSafeRootLength = 65
if ($Root.Length -gt $MaxSafeRootLength) {
    Write-Host ""
    Write-Host "[ERROR] The project folder path is too long for a reliable Windows install." -ForegroundColor Red
    Write-Host "Current path ($($Root.Length) characters): $Root" -ForegroundColor Yellow
    Write-Host "Move the project to a shorter path, for example:" -ForegroundColor Yellow
    Write-Host "  C:\Zundanen" -ForegroundColor Cyan
    Write-Host "or:" -ForegroundColor Yellow
    Write-Host "  C:\Zundanen" -ForegroundColor Cyan
    Write-Host "Then run setup.bat again." -ForegroundColor Yellow
    throw "Project path is too long. Use a project path of $MaxSafeRootLength characters or fewer."
}

if ($Force -and (Test-Path $Runtime)) {
    Write-Warning "-Force specified: removing the existing runtime folder."
    Remove-Item $Runtime -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $Runtime, $TtsRoot, $RvcRoot, $Exports | Out-Null

Write-Step "1 / 8  Installing Windows prerequisites"
if (-not (Get-Command winget.exe -ErrorAction SilentlyContinue)) {
    throw "WinGet was not found. Install/repair 'App Installer' from Microsoft Store, then run setup.bat again."
}
# PyTorch DLLs require the Microsoft Visual C++ runtime on a clean Windows install.
# Skip installation when a usable x64 runtime is already present.
Ensure-VCRedistX64
Ensure-WebView2Runtime
Ensure-WingetPackage "Git.Git"
Ensure-WingetPackage "Gyan.FFmpeg"
Ensure-WingetPackage "Python.Python.3.11"
Ensure-WingetPackage "Python.Python.3.12"

$Git = Resolve-Git
$Py311 = Resolve-Python "3.11" "311"
$Py312 = Resolve-Python "3.12" "312"
Write-Host "Git        : $Git"
Write-Host "Python 3.11: $Py311"
Write-Host "Python 3.12: $Py312"

$gpu = Get-GpuInfo
Write-Host "GPU        : $($gpu.Name)"
if ($gpu.Capability -gt 0) { Write-Host "Compute cap: $($gpu.Capability)" }
if ($gpu.Mode -eq "cpu") {
    Write-Warning "No NVIDIA GPU detected. Voice Generation can fall back to CPU, but Model Trainer currently requires NVIDIA CUDA."
}

Write-Step "2 / 8  Preparing Finnish Chatterbox source"
if (-not (Test-Path (Join-Path $TtsRepo ".git"))) {
    if (Test-Path $TtsRepo) { Remove-Item $TtsRepo -Recurse -Force }
    $oldSkip = $env:GIT_LFS_SKIP_SMUDGE
    $env:GIT_LFS_SKIP_SMUDGE = "1"
    try {
        Invoke-Native $Git @("clone", "--depth", "1", "https://huggingface.co/Finnish-NLP/Chatterbox-Finnish", $TtsRepo)
    }
    finally {
        if ($null -eq $oldSkip) { Remove-Item Env:GIT_LFS_SKIP_SMUDGE -ErrorAction SilentlyContinue }
        else { $env:GIT_LFS_SKIP_SMUDGE = $oldSkip }
    }
} else {
    Write-Host "Chatterbox-Finnish source already exists; keeping it."
}

Write-Step "3 / 8  Creating Finnish TTS environment"
if (-not (Test-Path (Join-Path $TtsVenv "Scripts\python.exe"))) {
    Invoke-Native $Py311 @("-m", "venv", $TtsVenv)
}
$TtsPython = Join-Path $TtsVenv "Scripts\python.exe"
Invoke-Native $TtsPython @("-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools==80.9.0")

if ($gpu.Mode -eq "blackwell") {
    Invoke-Native $TtsPython @("-m", "pip", "install", "torch==2.10.0", "torchvision==0.25.0", "torchaudio==2.10.0", "--index-url", "https://download.pytorch.org/whl/cu128")
    Invoke-Native $TtsPython @("-m", "pip", "install", "xformers==0.0.35", "--index-url", "https://download.pytorch.org/whl/cu128")
} elseif ($gpu.Mode -eq "nvidia") {
    Invoke-Native $TtsPython @("-m", "pip", "install", "torch==2.5.1", "torchvision==0.20.1", "torchaudio==2.5.1", "--index-url", "https://download.pytorch.org/whl/cu124")
    Invoke-Native $TtsPython @("-m", "pip", "install", "xformers==0.0.28.post3", "--index-url", "https://download.pytorch.org/whl/cu124")
} else {
    Invoke-Native $TtsPython @("-m", "pip", "install", "torch==2.5.1", "torchvision==0.20.1", "torchaudio==2.5.1", "--index-url", "https://download.pytorch.org/whl/cpu")
}

$TtsRequirements = Join-Path $TtsRoot "requirements.windows.txt"
Get-Content (Join-Path $TtsRepo "requirements.txt") |
    Where-Object { $_ -notmatch '^\s*torchao\s*==' } |
    Set-Content -Encoding UTF8 $TtsRequirements
Invoke-Native $TtsPython @("-m", "pip", "install", "-r", $TtsRequirements)
Invoke-Native $TtsPython @("-m", "pip", "install", "huggingface_hub>=0.26,<1.0", "flask>=3.0,<4", "pywebview==6.2.1")
Invoke-Native $TtsPython @("-m", "pip", "install", "setuptools==80.9.0")

Write-Step "4 / 8  Downloading Finnish TTS weights"
$TtsHf = Join-Path $TtsVenv "Scripts\hf.exe"
if (-not (Test-Path $TtsHf)) { throw "hf.exe was not installed in the TTS environment." }
$FineTuneWeight = Join-Path $TtsRepo "models\best_finnish_multilingual_cp986.safetensors"
$ReferenceAudio = Join-Path $TtsRepo "samples\reference_finnish.wav"
$needWeight = (-not (Test-Path $FineTuneWeight)) -or ((Get-Item $FineTuneWeight).Length -lt 10000000)
$needReference = (-not (Test-Path $ReferenceAudio)) -or ((Get-Item $ReferenceAudio).Length -lt 1024)
if ($needWeight) {
    Remove-Item $FineTuneWeight -Force -ErrorAction SilentlyContinue
    Invoke-Native $TtsHf @("download", "Finnish-NLP/Chatterbox-Finnish", "models/best_finnish_multilingual_cp986.safetensors", "--revision", "main", "--local-dir", $TtsRepo)
} else {
    Write-Host "Finnish fine-tune weight already downloaded; keeping it."
}
if ($needReference) {
    Remove-Item $ReferenceAudio -Force -ErrorAction SilentlyContinue
    Invoke-Native $TtsHf @("download", "Finnish-NLP/Chatterbox-Finnish", "samples/reference_finnish.wav", "--revision", "main", "--local-dir", $TtsRepo)
} else {
    Write-Host "Finnish reference audio already downloaded; keeping it."
}
Invoke-Native $TtsPython @("setup.py") $TtsRepo

Write-Step "5 / 8  Preparing RVC source"
if (-not (Test-Path (Join-Path $RvcRepo ".git"))) {
    if (Test-Path $RvcRepo) { Remove-Item $RvcRepo -Recurse -Force }
    Invoke-Native $Git @("clone", "--depth", "1", "https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git", $RvcRepo)
} else {
    Write-Host "RVC source already exists; keeping it."
}

# Make RVC honor the explicit language selected by Zundanen instead of
# always inheriting the Windows display language through locale.getdefaultlocale().
Invoke-Native $TtsPython @("-c", "import sys; sys.path.insert(0, r'$Root'); from rvc_locale import ensure_rvc_language_override; print('RVC locale patch applied:', ensure_rvc_language_override(r'$RvcRepo'))")

Write-Step "6 / 8  Creating RVC environment"
if (-not (Test-Path (Join-Path $RvcVenv "Scripts\python.exe"))) {
    Invoke-Native $Py312 @("-m", "venv", $RvcVenv)
}
$RvcPython = Join-Path $RvcVenv "Scripts\python.exe"
Invoke-Native $RvcPython @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")

if ($gpu.Mode -eq "blackwell") {
    $RvcReqName = "requirments_cu128_py312.txt"
    Invoke-Native $RvcPython @("-m", "pip", "install", "torch==2.7.1+cu128", "torchaudio==2.7.1+cu128", "--index-url", "https://download.pytorch.org/whl/cu128", "--extra-index-url", "https://pypi.org/simple")
} elseif ($gpu.Mode -eq "nvidia") {
    $RvcReqName = "requirments_cu118_py312.txt"
    Invoke-Native $RvcPython @("-m", "pip", "install", "torch==2.7.1+cu118", "torchaudio==2.7.1+cu118", "--index-url", "https://download.pytorch.org/whl/cu118", "--extra-index-url", "https://pypi.org/simple")
} else {
    $RvcReqName = "requirments_cpu_py312.txt"
}

$RvcReqSource = Join-Path $RvcRepo $RvcReqName
$RvcReqOfficial = Join-Path $RvcRoot "requirements.official.txt"
$content = Get-Content $RvcReqSource -Raw
$content = $content.Replace("https://mirrors.pku.edu.cn/pypi/simple", "https://pypi.org/simple")
$content = $content.Replace("https://mirrors.nju.edu.cn/pytorch/whl/cpu", "https://download.pytorch.org/whl/cpu")
$content = $content.Replace("https://mirrors.nju.edu.cn/pytorch/whl/cu118", "https://download.pytorch.org/whl/cu118")
$content = $content.Replace("https://mirrors.nju.edu.cn/pytorch/whl/cu128", "https://download.pytorch.org/whl/cu128")
Set-Content -Path $RvcReqOfficial -Value $content -Encoding UTF8
Invoke-Native $RvcPython @("-m", "pip", "install", "-r", $RvcReqOfficial)
Invoke-Native $RvcPython @("-m", "pip", "install", "huggingface_hub>=0.26,<1.0")

Write-Step "7 / 8  Downloading RVC base assets"
$RvcHf = Join-Path $RvcVenv "Scripts\hf.exe"
if (-not (Test-Path $RvcHf)) { throw "hf.exe was not installed in the RVC environment." }
Invoke-Native $RvcHf @("download", "lj1995/VoiceConversionWebUI", "--revision", "main", "--include", "hubert_base/*", "--local-dir", (Join-Path $RvcRepo "assets"))
Invoke-Native $RvcHf @("download", "lj1995/VoiceConversionWebUI", "rmvpe.pt", "--revision", "main", "--local-dir", (Join-Path $RvcRepo "assets\rmvpe"))
Invoke-Native $RvcHf @("download", "lj1995/VoiceConversionWebUI", "--revision", "main", "--include", "pretrained/*", "pretrained_v2/*", "--local-dir", (Join-Path $RvcRepo "assets"))
$ModelDownloads = Join-Path $RvcRepo ".model-downloads"
New-Item -ItemType Directory -Force -Path $ModelDownloads | Out-Null
Invoke-Native $RvcHf @("download", "lj1995/VoiceConversionWebUI", "mute.zip", "--revision", "main", "--local-dir", $ModelDownloads)
$MuteZip = Join-Path $ModelDownloads "mute.zip"
$MuteDest = Join-Path $RvcRepo "logs"
New-Item -ItemType Directory -Force -Path $MuteDest | Out-Null
Invoke-Native $RvcPython @("-m", "zipfile", "-e", $MuteZip, $MuteDest)
New-Item -ItemType Directory -Force -Path (Join-Path $RvcRepo "assets\weights"), (Join-Path $RvcRepo "assets\indices"), $Exports | Out-Null

Write-Step "8 / 8  Verifying installation"
Invoke-Native $TtsPython @("-c", "import torch, flask, soundfile, safetensors; print('TTS torch:', torch.__version__); print('TTS CUDA:', torch.cuda.is_available())")
Invoke-Native $RvcPython @("-c", "import torch, faiss, librosa, numpy; print('RVC torch:', torch.__version__); print('RVC CUDA:', torch.cuda.is_available()); print('NumPy:', numpy.__version__); print('FAISS: OK')")
Invoke-Native (Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe") @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", (Join-Path $Root "doctor.ps1"))

Set-Content -Path (Join-Path $Runtime ".setup-complete") -Value (Get-Date -Format o) -Encoding UTF8
Write-Host ""
Write-Host "SETUP COMPLETE" -ForegroundColor Green
Write-Host "Run: Zundanen.exe"
}
finally {
    Stop-SetupSleepPrevention
}
