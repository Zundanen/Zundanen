Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
try {
    [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
    $OutputEncoding = [Console]::OutputEncoding
} catch {}
$ProgressPreference = "SilentlyContinue"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Runtime = Join-Path $Root "runtime"
$RvcPython = Join-Path $Runtime "rvc\.venv\Scripts\python.exe"
$StatusPath = Join-Path $Runtime "accent_support_status.json"
$DefaultNativeRuntimeUrl = "https://github.com/Zundanen/Zundanen/releases/download/v1.1.0/Zundanen-Aalto-Native-v1.1.0.zip"
$NativeRuntimeUrl = if ($env:ZUNDANEN_AALTO_NATIVE_URL) { $env:ZUNDANEN_AALTO_NATIVE_URL } else { $DefaultNativeRuntimeUrl }
$script:LastNativeExitCode = 0

function Write-Status([string]$State, [string]$Message) {
    New-Item -ItemType Directory -Force -Path $Runtime | Out-Null
    @{
        state = $State
        message = $Message
        reboot_required = $false
        updated_at = (Get-Date -Format o)
    } | ConvertTo-Json | Set-Content -Path $StatusPath -Encoding UTF8
}

function Refresh-Path {
    $machine = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $user = [Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = "$machine;$user"
}

function Invoke-NativeCommand {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [string[]]$Arguments = @(),
        [switch]$Quiet
    )

    $oldPreference = $ErrorActionPreference
    $script:LastNativeExitCode = -1
    $ErrorActionPreference = "Continue"
    try {
        if ($Quiet) {
            & $FilePath @Arguments *> $null
        } else {
            & $FilePath @Arguments 2>&1 | ForEach-Object { Write-Host $_ }
        }
        $script:LastNativeExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $oldPreference
    }
}

function Install-NativeAaltoRuntime([string]$Source) {
    $dest = Join-Path $Runtime "aalto_native"
    $zip = Join-Path $env:TEMP "Zundanen-Aalto-Native-v1.1.0.zip"
    Remove-Item -Force $zip -ErrorAction SilentlyContinue

    Write-Status "downloading_native" "Downloading the Windows-native Aalto/Kaldi runtime."
    Write-Host "Downloading Windows-native Aalto/Kaldi runtime..." -ForegroundColor Yellow

    if ($Source -match '^(?i)https?://') {
        try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 } catch {}
        Invoke-WebRequest -UseBasicParsing -Uri $Source -OutFile $zip
    } elseif (Test-Path $Source) {
        Copy-Item -Force $Source $zip
    } else {
        throw "Native Aalto runtime source was not found: $Source"
    }

    if (-not (Test-Path $zip)) {
        throw "Native Aalto runtime download did not create a ZIP file."
    }

    Remove-Item -Recurse -Force $dest -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    Expand-Archive -Path $zip -DestinationPath $dest -Force
    Remove-Item -Force $zip -ErrorAction SilentlyContinue
}

try {
    Write-Host "Zundanen v2.0.0 - Accent components" -ForegroundColor Green
    Write-Host "These components are normally installed automatically by setup.bat."
    Write-Status "starting" "Checking Accent components."

    if (-not (Test-Path $RvcPython)) {
        throw "The Zundanen RVC environment was not found. Run setup.bat first."
    }

    Write-Host ""
    Write-Host "[1/3] PyWORLD" -ForegroundColor Cyan
    Invoke-NativeCommand -FilePath $RvcPython -Arguments @(
        "-c", "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('pyworld') else 1)"
    ) -Quiet

    if ($script:LastNativeExitCode -ne 0) {
        Write-Status "installing_pyworld" "Installing PyWORLD."
        Invoke-NativeCommand -FilePath $RvcPython -Arguments @(
            "-m", "pip", "install", "--disable-pip-version-check", "pyworld==0.3.5"
        )
        if ($script:LastNativeExitCode -ne 0) {
            throw "PyWORLD installation failed (exit code $($script:LastNativeExitCode))."
        }
    }

    Invoke-NativeCommand -FilePath $RvcPython -Arguments @(
        "-c", "import pyworld; print('[OK] PyWORLD', pyworld.__version__)"
    )
    if ($script:LastNativeExitCode -ne 0) {
        throw "PyWORLD could not be imported (exit code $($script:LastNativeExitCode))."
    }

    $nativeProbe = Join-Path $Root "aalto_native_probe.py"
    if (-not (Test-Path $nativeProbe)) {
        throw "aalto_native_probe.py is missing from the Zundanen package."
    }

    Write-Host ""
    Write-Host "[2/3] Windows-native Aalto/Kaldi runtime" -ForegroundColor Cyan
    Invoke-NativeCommand -FilePath $RvcPython -Arguments @($nativeProbe, "--files-only") -Quiet
    if ($script:LastNativeExitCode -ne 0) {
        Install-NativeAaltoRuntime -Source $NativeRuntimeUrl
        Invoke-NativeCommand -FilePath $RvcPython -Arguments @($nativeProbe, "--files-only") -Quiet
        if ($script:LastNativeExitCode -ne 0) {
            throw "The native Aalto runtime was downloaded, but required files are still missing."
        }
    }
    Write-Host "[OK] Native Aalto/Kaldi files are present." -ForegroundColor Green

    Write-Host ""
    Write-Host "[3/3] Native runtime verification" -ForegroundColor Cyan
    Invoke-NativeCommand -FilePath $RvcPython -Arguments @($nativeProbe) -Quiet

    if ($script:LastNativeExitCode -ne 0) {
        Write-Status "installing_mkl" "Installing Intel oneMKL for the native Aalto backend."
        $winget = Get-Command winget.exe -ErrorAction SilentlyContinue
        if (-not $winget) {
            throw "The native Aalto files are present, but a runtime dependency is missing and WinGet was not found. Install/repair App Installer, then run setup.bat again."
        }

        Write-Host "Installing/updating Intel oneMKL..." -ForegroundColor Yellow
        Invoke-NativeCommand -FilePath $winget.Source -Arguments @(
            "install", "--id", "Intel.oneMKL", "-e", "--source", "winget",
            "--accept-package-agreements", "--accept-source-agreements",
            "--silent", "--disable-interactivity"
        ) -Quiet
        $installCode = $script:LastNativeExitCode

        if ($installCode -ne 0) {
            Invoke-NativeCommand -FilePath $winget.Source -Arguments @(
                "list", "--id", "Intel.oneMKL", "-e",
                "--accept-source-agreements", "--disable-interactivity"
            ) -Quiet
            if ($script:LastNativeExitCode -ne 0) {
                throw "Intel oneMKL installation failed (WinGet exit code $installCode)."
            }
        }

        Write-Host "[OK] Intel oneMKL is installed." -ForegroundColor Green
        Refresh-Path
        Invoke-NativeCommand -FilePath $RvcPython -Arguments @($nativeProbe) -Quiet
    }

    if ($script:LastNativeExitCode -ne 0) {
        throw "The Windows-native Aalto/Kaldi runtime could not start. Run aalto_native_probe.py for details."
    }

    Write-Status "complete" "Accent components are ready."
    Write-Host ""
    Write-Host "[OK] ACCENT READY" -ForegroundColor Green
    Write-Host "WSL2 and Docker Desktop are not required." -ForegroundColor Green
    exit 0
}
catch {
    $message = $_.Exception.Message
    Write-Status "error" $message
    Write-Host ""
    Write-Host "[ERROR] $message" -ForegroundColor Red
    exit 1
}
